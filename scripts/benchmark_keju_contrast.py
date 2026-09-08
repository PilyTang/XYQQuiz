"""Compare the fixed small-text contrast experiment on a private Keju corpus."""
from pathlib import Path
import argparse
import dataclasses
import hashlib
import json
import time

import cv2
import numpy as np
from xyq_quiz.capture.models import CapturedFrame
from xyq_quiz.config import AppConfig
from xyq_quiz.knowledge.matcher import QuestionMatcher
from xyq_quiz.knowledge.updater import load_current_generation
from xyq_quiz.recognition.layout import LayoutProfile, build_layout_detector
from xyq_quiz.recognition.ocr import RapidOCREngine
from xyq_quiz.recognition.pipeline import RecognitionPipeline


def enhance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = np.clip((gray.astype(np.float32)-40)*255/130, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class ContrastOCR(RapidOCREngine):
    def _recognize_line(self, image):
        return super()._recognize_line(enhance(image))

    def recognize(self, image):
        return super().recognize(enhance(image))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=3)
    args=parser.parse_args()
    if args.rounds < 1:parser.error('--rounds must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=args.corpus/'tests/fixtures/recognition/manifest.json'
    cases=[]
    for row in json.loads(manifest.read_text(encoding='utf-8'))['cases']:
        cases.append((manifest.parent/row['file'],row))
    diagnostics=args.corpus/'diagnostics'
    for path in sorted(diagnostics.glob('*.jpg')):
        if path.name.startswith('live-keju-'):cases.append((path,{}))
    for path in sorted(diagnostics.glob('*/frame.jpg')):cases.append((path,{}))
    for path in sorted(diagnostics.glob('wgc-preview-*.png')):cases.append((path,{}))
    root=Path(__file__).resolve().parents[1]
    config=AppConfig.load(root/'config.example.json')
    bank=load_current_generation(root/'data').question_bank
    pipelines={}
    for mode,engine in [('original',RapidOCREngine()),('contrast',ContrastOCR())]:
        detector=build_layout_detector([LayoutProfile.load(p) for p in config.effective_layout_paths])
        pipelines[mode]=RecognitionPipeline(detector,engine,QuestionMatcher(bank,92,5,90))
        pipelines[mode].warm_up()
    seen=set(); results=[]
    try:
        for path,truth in cases:
            data=path.read_bytes(); digest=hashlib.sha256(data).hexdigest()
            if digest in seen:continue
            seen.add(digest)
            image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
            item={'sample':len(results)+1,'file':str(path),'sha256':digest,'size':[image.shape[1],image.shape[0]],'truth':truth,'results':{}}
            for mode,pipeline in (list(pipelines.items()) if item['sample']%2 else list(pipelines.items())[::-1]):
                samples=[]
                for repeat in range(args.rounds):
                    result=pipeline.recognize(CapturedFrame.create(repeat+1,time.monotonic_ns(),image),repeat+1)
                    samples.append(result.timings.ocr_ms)
                item['results'][mode]=dataclasses.asdict(result)
                item['results'][mode]['ocr_ms_samples']=samples
                if mode=='original':
                    folder=args.output/f"sample-{item['sample']:02d}";folder.mkdir(exist_ok=True)
                    for idx,crop in enumerate(pipeline.latest_crops()):
                        cv2.imencode('.png',crop)[1].tofile(folder/f'crop-{idx}.png')
            results.append(item)
            print(json.dumps({'sample':item['sample'],'name':path.name,'truth':truth.get('expected_option_index'),'original':{k:item['results']['original'][k] for k in ['source_id','option_index','question_text','option_texts']},'contrast':{k:item['results']['contrast'][k] for k in ['source_id','option_index','question_text','option_texts']}},ensure_ascii=False),flush=True)
    finally:
        for p in pipelines.values():p.close()
    (args.output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
