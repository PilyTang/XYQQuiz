from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
from types import SimpleNamespace
from zipfile import ZipFile

from xyq_quiz.performance_recording import PerformanceRecording, option_call, note_ocr_fallback
from xyq_quiz.recognition.models import ActivityKind, ConfidenceLevel, RecognitionTimings


def test_fresh_launcher_import_has_no_recording_backend_cycle():
    completed = subprocess.run([sys.executable,'-c','import xyq_quiz.launcher'],
                               capture_output=True,text=True,timeout=20)
    assert completed.returncode == 0, completed.stderr


def test_recording_times_threaded_options_and_exports_no_content(tmp_path, monkeypatch):
    from xyq_quiz import performance_recording as module
    clock = [1_000_000_000]
    monkeypatch.setattr(module.time,'perf_counter_ns',lambda: clock[0])
    recorder = PerformanceRecording()
    recorder.observe(clock[0],clock[0],2,(1024,768))
    assert recorder.report()['questions']==[]
    recorder.start({'ocr':'directml:0','preview':'windows_hardware:0'})
    recorder.observe(clock[0],clock[0]+1_000_000,2,(1024,768))
    clock[0]+=100_000_000
    recorder.begin(1,ActivityKind.TEACHERS_DAY)
    def recognize():
        def ocr():
            note_ocr_fallback()
            clock[0]+=10_000_000
            return 'PRIVATE OPTION TEXT'
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(option_call(0,ocr)).result()=='PRIVATE OPTION TEXT'
        return SimpleNamespace(timings=RecognitionTimings(2,10,4,16),confidence_level=ConfidenceLevel.HIGH)
    recorder.execute(1,recognize)
    clock[0]+=4_000_000
    recorder.complete(1,3,True,ConfidenceLevel.HIGH)
    trace = recorder.sent(1,3,{'ocr':'directml:0'})
    clock[0]+=5_000_000
    assert not recorder.acknowledge(trace['session'],1,2,'received',1)
    assert recorder.acknowledge(trace['session'],1,3,'received',1)
    recorder.stop()
    report = recorder.report()
    row=report['questions'][0]
    assert row['capture_to_answer_ms']==114
    assert row['stability_wait_ms']==97
    assert row['capture_to_received_ack_ms']==119
    assert row['attempts'][0]['options'][0]['elapsed_ms']==10
    assert row['attempts'][0]['options'][0]['fallbacks']==1
    assert report['summary']['capture_to_answer_ms']['p95_ms']==114
    with ZipFile(recorder.export(tmp_path)) as archive:
        assert set(archive.namelist())=={'report.json','summary.csv','README.txt'}
        content=archive.read('report.json').decode()
        assert 'PRIVATE' not in content
        assert 'captured_ns' not in content
        assert json.loads(content)['questions'][0]['attempts'][0]['started_at_ms']==100
    recorder.start()
    recorder.begin(1,ActivityKind.TEACHERS_DAY)
    assert not recorder.acknowledge(trace['session'],1,3,'received',1)


def test_recording_bounds_cache_and_missing_stages(tmp_path):
    recorder=PerformanceRecording()
    recorder.MAX_QUESTIONS=2
    recorder.start()
    recorder.begin(1,ActivityKind.KEJU)
    recorder.complete(1,3,True,ConfidenceLevel.HIGH)
    recorder.begin(2,ActivityKind.KEJU)
    recorder.begin(3,ActivityKind.KEJU)
    assert not recorder.enabled
    report=recorder.report()
    assert len(report['questions'])==2
    assert report['questions'][0]['cache_hit']
    assert 'capture_to_answer_ms' not in report['questions'][0]
    assert report['questions'][1]['outcome']=='pending'
    assert report['summary']['capture_to_answer_ms'] is None


def test_retry_attempts_do_not_become_new_questions():
    recorder=PerformanceRecording()
    recorder.start()
    recorder.begin(1,ActivityKind.TEACHERS_DAY)
    for level in (ConfidenceLevel.NONE,ConfidenceLevel.HIGH):
        recorder.execute(1,lambda: SimpleNamespace(timings=RecognitionTimings(1,2,3,6),confidence_level=level))
        recorder.complete(1,2 if level==ConfidenceLevel.NONE else 3,level==ConfidenceLevel.HIGH,level)
    report=recorder.report()
    assert report['status']['questions']==1
    assert report['status']['attempts']==2
    assert report['questions'][0]['outcome']=='HIGH'
    assert not report['questions'][0]['cache_hit']


def test_stopped_recording_does_not_instrument_new_work():
    recorder=PerformanceRecording()
    recorder.start()
    recorder.begin(1,ActivityKind.KEJU)
    recorder.stop()
    assert recorder.execute(1,lambda: 'unmodified')=='unmodified'
    assert recorder.report()['status']['attempts']==0
