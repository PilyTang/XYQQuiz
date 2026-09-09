"""Build a reviewed supplement generation from locally downloaded source bytes."""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from xyq_quiz.knowledge.teacher_bank import decode_icon, load_teacher_bank, TeacherSkillRecord


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--downloads',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]/'data/teachers_day/supplements'
    old=load_teacher_bank(root)
    assert old.count==18, 'Import starts from the retained 18-record generation'
    entries=json.loads((args.downloads/'download.json').read_text(encoding='utf-8'))
    selected=[entry for entry in entries if 'error' not in entry]
    assert len(selected)==13
    generation='20260910-yzz-31'
    target=root/'generations'/generation
    target.mkdir()
    (target/'icons').mkdir()
    records=list(old.records)
    for record in old.records:
        shutil.copy2(old.directory/record.image_path,target/record.image_path)
    for entry in selected:
        data=(args.downloads/entry['file']).read_bytes()
        decode_icon(data)
        digest=hashlib.sha256(data).hexdigest()
        assert digest==entry['sha256']
        name=entry['name']
        # The September table has a typo; January skill documentation uses 结.
        if name=='广积善缘': name='广结善缘'
        source_url=entry['url'].replace('https://','http://')
        record=TeacherSkillRecord('teachers_day:yzz:'+hashlib.sha256(source_url.encode()).hexdigest()[:16],
            name,'补充技能','supplement',source_url,'icons/'+digest+'.png',digest)
        records.append(record)
        (target/record.image_path).write_bytes(data)
    encoded=(json.dumps([asdict(r) for r in records],ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    (target/'records.json').write_bytes(encoded)
    metadata=dict(schema_version=1,generation_id=generation,source_url='https://xyq.yzz.cn/focus/202608/1768536.shtml',
                  updated_at='2026-09-10',record_count=len(records),image_count=len(records),
                  records_sha256=hashlib.sha256(encoded).hexdigest())
    (target/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (root/'current.json').write_text(json.dumps(dict(schema_version=1,generation_id=generation),indent=2)+'\n',encoding='utf-8')
    print('Verified supplement records:',load_teacher_bank(root).count)


if __name__=='__main__': main()
