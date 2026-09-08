from pathlib import Path
import json
import time
import cv2
import numpy as np
from xyq_quiz.recognition.ocr import RapidOCREngine, OCRRole, segment_text_lines

root=Path(__file__).resolve().parents[1] / 'tests/fixtures/teachers_day'
cases=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
engine=RapidOCREngine()
results=[]
for case in cases:
    for idx, key in enumerate('abcd'):
        image=cv2.imdecode(np.frombuffer((root/case['files'][f'option-{key}']).read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        # The fixture contains the whole button; role segmentation removes its edges.
        try: lines=segment_text_lines(image,OCRRole.OPTION)
        except Exception: lines=(image,)
        variants={'original':lines}
        gray=[cv2.cvtColor(x,cv2.COLOR_BGR2GRAY) for x in lines]
        for threshold in (90,120,150):
            variants[f'mask{threshold}']=[cv2.cvtColor(np.where(x<threshold,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR) for x in gray]
        variants['contrast']=[cv2.cvtColor(np.clip((x.astype(float)-40)*255/130,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR) for x in gray]
        scores={}
        for label,items in variants.items():
            start=time.perf_counter(); outputs=[engine._recognize_line(x) for x in items]
            scores[label]={'text':''.join(x.text for x in outputs),'confidence':min(x.confidence for x in outputs),'ms':(time.perf_counter()-start)*1000}
        results.append({'sample':case['sample'],'option':key,'expected':case['options'][idx],'results':scores})
summary={v:{'exact':sum(r['results'][v]['text']==r['expected'] for r in results),'total':len(results)} for v in results[0]['results']}
out=Path('build/small-text-benchmark.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'summary':summary,'cases':results},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'summary':summary,'differences':[r for r in results if len({x['text'] for x in r['results'].values()})>1]},ensure_ascii=False,indent=2))
