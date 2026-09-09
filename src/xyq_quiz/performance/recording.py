"""Opt-in, bounded, content-free latency recording. All server times use QPC."""
from __future__ import annotations

from collections import OrderedDict
from contextvars import ContextVar
import copy
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import statistics
import threading
import time
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED

from xyq_quiz import __version__

_CURRENT = ContextVar('performance_attempt', default=None)


def option_call(index, callback, *args):
    """Capture context before crossing the pipeline's OCR worker boundary."""
    context = _CURRENT.get()
    queued = time.perf_counter_ns() if context else 0
    def call():
        if context is None:
            return callback(*args)
        recorder, session, generation, attempt, _ = context
        token = _CURRENT.set((recorder, session, generation, attempt, index))
        started = time.perf_counter_ns()
        try:
            return callback(*args)
        finally:
            recorder.option(session, generation, attempt, index, queued, started, time.perf_counter_ns())
            _CURRENT.reset(token)
    return call


def note_ocr_fallback():
    context = _CURRENT.get()
    if context is not None:
        recorder, session, generation, attempt, index = context
        recorder.fallback(session, generation, attempt, index)


class PerformanceRecording:
    MAX_QUESTIONS = 500
    MAX_ATTEMPTS = 2000

    def __init__(self):
        self._lock = threading.RLock()
        self.enabled = False
        self.session = ''
        self._rows = OrderedDict()
        self._pending = None
        self._attempt_count = 0
        self._metadata = {}
        self._started = None
        self._stop_reason = None

    def start(self, metadata=None):
        with self._lock:
            if self.enabled:
                return self.status()
            self.session = uuid4().hex
            self._rows.clear()
            self._pending = None
            self._attempt_count = 0
            self._metadata = metadata or {}
            self._started = datetime.now(timezone.utc).isoformat()
            self._stop_reason = None
            self.enabled = True
            return self.status()

    def stop(self):
        with self._lock:
            self.enabled = False
            self._stop_reason = self._stop_reason or 'manual'
            return self.status()

    def status(self):
        with self._lock:
            return dict(enabled=self.enabled, session=self.session, questions=len(self._rows),
                        attempts=self._attempt_count, stop_reason=self._stop_reason)

    def observe(self, captured_ns, layout_started_ns, layout_ms, frame_size, entry_kind='panel_or_reset'):
        if not self.enabled:
            return
        with self._lock:
            self._pending = dict(captured_ns=captured_ns, layout_started_ns=layout_started_ns,
                                 first_layout_ms=layout_ms, frame_size=list(frame_size), entry_kind=entry_kind)

    def begin(self, generation, activity):
        if not self.enabled:
            return
        with self._lock:
            if len(self._rows) >= self.MAX_QUESTIONS:
                self.enabled = False
                self._stop_reason = 'question_limit'
                return
            now = time.perf_counter_ns()
            row = dict(self._pending or {})
            row.update(question=len(self._rows)+1, activity=activity.value,
                       stable_ns=now, attempts=[], cache_hit=False, outcome='pending')
            self._pending = None
            self._rows[generation] = row

    def execute(self, generation, callback, *args):
        with self._lock:
            row = self._rows.get(generation) if self.enabled else None
            if row is None:
                context = None
            elif self._attempt_count >= self.MAX_ATTEMPTS:
                self.enabled = False
                self._stop_reason = 'attempt_limit'
                context = None
            else:
                self._attempt_count += 1
                attempt = dict(started_ns=time.perf_counter_ns(), options=[], fallbacks=0)
                row['attempts'].append(attempt)
                context = (self, self.session, generation, len(row['attempts'])-1, None)
        if context is None:
            return callback(*args)
        token = _CURRENT.set(context)
        try:
            result = callback(*args)
            with self._lock:
                if self.session == context[1]:
                    attempt.update(ocr_ms=result.timings.ocr_ms, match_ms=result.timings.match_ms,
                                   layout_ms=result.timings.layout_ms, level=result.confidence_level.value)
            return result
        finally:
            with self._lock:
                if self.session == context[1]:
                    attempt['finished_ns'] = time.perf_counter_ns()
            _CURRENT.reset(token)

    def _attempt(self, session, generation, index):
        row = self._rows.get(generation)
        if session == self.session and row and index < len(row['attempts']):
            return row['attempts'][index]
        return None

    def option(self, session, generation, attempt, index, queued, started, finished):
        with self._lock:
            record = self._attempt(session, generation, attempt)
            if record is not None:
                record['options'].append(dict(index=index, queue_ms=(started-queued)/1e6,
                                               elapsed_ms=(finished-started)/1e6,
                                               fallbacks=record.get('option_fallbacks',{}).get(str(index),0)))

    def fallback(self, session, generation, attempt, index):
        with self._lock:
            record = self._attempt(session, generation, attempt)
            if record is not None:
                record['fallbacks'] += 1
                counts = record.setdefault('option_fallbacks',{})
                counts[str(index)] = counts.get(str(index),0) + 1

    def complete(self, generation, version, drawable, level):
        with self._lock:
            row = self._rows.get(generation) if self.enabled else None
            if row is None:
                return
            row['outcome'] = level.value
            if row['attempts']:
                row['attempts'][-1]['accepted_ns'] = time.perf_counter_ns()
            if drawable and 'result_ready_ns' not in row:
                row.update(result_ready_ns=time.perf_counter_ns(), answer_version=version,
                           cache_hit=not bool(row['attempts']))

    def sent(self, generation, version, backend):
        with self._lock:
            row = self._rows.get(generation) if self.enabled else None
            if row is None or row.get('answer_version') != version:
                return None
            row.setdefault('sent_ns', time.perf_counter_ns())
            row.setdefault('backend', backend)
            return dict(session=self.session, generation=generation, version=version)

    def acknowledge(self, session, generation, version, stage, elapsed_ms):
        with self._lock:
            row = self._rows.get(generation)
            if (session != self.session or row is None or row.get('answer_version') != version
                    or 'sent_ns' not in row):
                return False
            row.setdefault('ui', {}).setdefault(stage, dict(
                acknowledged_ns=time.perf_counter_ns(), client_elapsed_ms=elapsed_ms))
            return True

    def report(self):
        with self._lock:
            rows = copy.deepcopy(list(self._rows.values()))
            status = self.status()
            metadata = copy.deepcopy(self._metadata)
            started = self._started
        for row in rows:
            start = row.get('captured_ns', row['stable_ns'])
            if 'captured_ns' in row:
                row['capture_to_stable_ms'] = (row['stable_ns']-start)/1e6
                row['capture_to_layout_start_ms'] = (row['layout_started_ns']-start)/1e6
                row['stability_wait_ms'] = max(0., (row['stable_ns']-row['layout_started_ns'])/1e6-row['first_layout_ms'])
            if row['attempts']:
                row['first_queue_ms'] = (row['attempts'][0]['started_ns']-row['stable_ns'])/1e6
            if 'result_ready_ns' in row and 'captured_ns' in row:
                row['capture_to_answer_ms'] = (row['result_ready_ns']-start)/1e6
            if 'captured_ns' in row:
                for stage, value in row.get('ui', {}).items():
                    row[f'capture_to_{stage}_ack_ms'] = (value['acknowledged_ns']-start)/1e6
            # Export relative timings only, never pixel hashes or frame contents.
            def relative(item):
                if isinstance(item, dict):
                    return {(k[:-3]+'_at_ms' if k.endswith('_ns') else k):
                            (round((v-start)/1e6,3) if k.endswith('_ns') else relative(v))
                            for k,v in item.items()}
                if isinstance(item, list):
                    return [relative(x) for x in item]
                return round(item,3) if isinstance(item,float) else item
            row.update(relative(row))
            for key in list(row):
                if key.endswith('_ns'):
                    del row[key]
            row['attempts'] = relative(row['attempts'])
            row['ui'] = relative(row.get('ui',{}))
        metrics = ('capture_to_answer_ms','capture_to_stable_ms','stability_wait_ms',
                   'first_queue_ms','first_layout_ms','capture_to_received_ack_ms',
                   'capture_to_canvas_submitted_ack_ms','capture_to_native_command_ack_ms')
        def summary(values):
            values = sorted(values)
            return dict(count=len(values), median_ms=round(statistics.median(values),3),
                        p95_ms=round(values[math.ceil(.95*len(values))-1],3),max_ms=round(values[-1],3)) if values else None
        summaries = {key:summary([r[key] for r in rows if key in r]) for key in metrics}
        for key in ('ocr_ms','match_ms'):
            summaries[key] = summary([a[key] for r in rows for a in r['attempts'] if key in a])
        groups = {f'{activity}/{entry}': summary([r['capture_to_answer_ms'] for r in rows
                  if r['activity']==activity and r.get('entry_kind')==entry and 'capture_to_answer_ms' in r])
                  for activity,entry in {(r['activity'],r.get('entry_kind')) for r in rows}}
        return dict(schema_version=1,app_version=__version__,started_utc=started,status=status,
                    metadata=metadata,summary=summaries,answer_latency_groups=groups,questions=rows,
                    slowest_questions=[r['question'] for r in sorted(rows,key=lambda r:r.get('capture_to_answer_ms',-1),reverse=True)[:10]],
                    limitations=['Start is the first captured frame of the stable question, not the game render time.',
                                 'UI acknowledgements include return network/scheduling time.',
                                 'Canvas submitted and native command acknowledged do not prove monitor presentation.',
                                 'Missing stages are not zero. Layout-only unstable frames are not counted as questions.'])

    def export(self, directory: Path):
        report = self.report()
        if not report['questions']:
            raise ValueError('尚无题目记录；开启后请完成几道题再导出')
        directory.mkdir(parents=True,exist_ok=True)
        path = directory / f'xyq-quiz-performance-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}.zip'
        table = io.StringIO()
        writer = csv.writer(table)
        writer.writerow(['stage','count','median_ms','p95_ms','max_ms'])
        for key, value in report['summary'].items():
            if value:
                writer.writerow([key,value['count'],value['median_ms'],value['p95_ms'],value['max_ms']])
        with ZipFile(path,'x',ZIP_DEFLATED) as archive:
            archive.writestr('report.json',json.dumps(report,ensure_ascii=False,indent=2))
            archive.writestr('summary.csv','\ufeff'+table.getvalue())
            archive.writestr('README.txt','性能记录：report.json 含逐题、重试和选项耗时；summary.csv 含中位数、P95、最大值。\n起点是最终稳定题面的首张捕获帧；不含游戏渲染到捕获之前的等待。\nUI 回执含回程开销，canvas_submitted 为绘制提交，native_command 为原生画框指令已受理，均不代表屏幕实际呈现。\n未答出及未显示的记录仍保留，缺失值不等于零。没有截图、题目、选项文字、窗口名、路径或账号。')
        return path
