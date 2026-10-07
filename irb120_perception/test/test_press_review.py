import csv
import json
import numpy as np
from irb120_perception.press_review import write_review


def test_labels_survive_ranking_changes_and_new_options(tmp_path, monkeypatch):
    from irb120_perception import contact_validation
    monkeypatch.setattr(contact_validation, 'render', lambda *args: None)
    points=np.array([[0.,0,.2],[.006,0,.2],[.012,0,.2]])
    normals=np.tile([0.,0,1.],(3,1))
    rows=[dict(point=p.tolist(),decision='pass',torque_per_pull_N_m=float(3-i),
               views={'cam2':dict(usable=True)}) for i,p in enumerate(points)]
    report=dict(candidates=rows[:2],experiment='test',selected_point=points[0].tolist())
    write_review(tmp_path,report,points,normals,{})
    payload=json.loads((tmp_path/'press_review.json').read_text())
    first_id=payload['options'][0]['id']
    with (tmp_path/'press_feedback.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['option','judgment','reason'])
        writer.writerow([first_id,'preferred','centered'])
    rows[0]['torque_per_pull_N_m']=0
    report['candidates']=rows
    write_review(tmp_path,report,points,normals,{})
    with (tmp_path/'press_feedback.csv').open() as stream:
        feedback={r['option']:r for r in csv.DictReader(stream)}
    assert feedback[first_id]['judgment']=='preferred'
    payload=json.loads((tmp_path/'press_review.json').read_text())
    assert payload['options'][-1]['id']==first_id
    assert len(feedback)==4
    report['candidates']=[];report['selected_point']=None
    write_review(tmp_path,report,points,normals,{})
    assert 'abstain' in (tmp_path/'PRESS_REVIEW.md').read_text()
    assert json.loads((tmp_path/'press_review.json').read_text())['options']==[]
