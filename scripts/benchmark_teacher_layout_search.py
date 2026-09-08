"""Compare current teacher localization with a Git revision on private frames.

Example: python scripts/benchmark_teacher_layout_search.py --images PATH --output build/layout-benchmark.json
Input images and the per-file report should stay local.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time
import types

import cv2
import numpy as np
from xyq_quiz.recognition.teachers_day_layout import TeacherLayoutDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-ref', default='v0.4.3')
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=5)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('--rounds must be positive')
    root = Path(__file__).resolve().parents[1]
    source = subprocess.run(
        ['git', '-C', str(root), 'show', f'{args.baseline_ref}:src/xyq_quiz/recognition/teachers_day_layout.py'],
        check=True, capture_output=True, encoding='utf-8',
    ).stdout
    baseline = types.ModuleType('baseline_teacher_layout')
    exec(compile(source, '<baseline_teacher_layout>', 'exec'), baseline.__dict__)
    cv2.setNumThreads(2)
    rows = []
    for path in sorted(args.images.iterdir()):
        if path.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
            continue
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot decode {path.name}')
        row = {'sample': path.name, 'shape': image.shape[:2]}
        variants = [('baseline', baseline.TeacherLayoutDetector), ('optimized', TeacherLayoutDetector)]
        if len(rows) % 2:
            variants.reverse()
        for key, factory in variants:
            detector = factory(root / 'data/layouts/teachers-day.json')
            elapsed, cpu, outcomes = [], [], []
            for _ in range(args.rounds):
                detector._last = None  # Measure global reacquisition, not tracking.
                begin, begin_cpu = time.perf_counter(), time.process_time()
                layout = detector.detect(image)
                cpu.append((time.process_time()-begin_cpu)*1000)
                elapsed.append((time.perf_counter()-begin)*1000)
                outcomes.append(layout is not None)
            row[key] = {'median_ms': statistics.median(elapsed),
                        'median_cpu_ms': statistics.median(cpu),
                        'detected': outcomes, 'suspected': detector.suspected}
        rows.append(row)
    if not rows:
        parser.error('No input images found')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'Compared {len(rows)} frames; report: {args.output}')


if __name__ == '__main__':
    main()
