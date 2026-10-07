"""Small review queue for human labels; never applies a choice or runs motion."""
import csv
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree


def write_review(snapshot, report, points, normals, cameras):
    from .contact_validation import render, render_geometry
    if report.get('geometry'):
        render_geometry(snapshot, snapshot, report['geometry'], cameras)
    eligible = sorted((r for r in report['candidates'] if r['decision']=='pass'),
                      key=lambda r:-r['torque_per_pull_N_m'])
    options=[];tree=cKDTree(points)
    for row in eligible:
        point=np.asarray(row['point'])
        if any(np.linalg.norm(point-np.asarray(o['point'])) < .002 for o in options):
            continue
        normal=normals[tree.query(point)[1]]
        identity = json.dumps([row['point'], normal.tolist(), report.get('geometry')],
                              sort_keys=True, separators=(',', ':'))
        option_id = 'press_option_' + hashlib.sha256(identity.encode()).hexdigest()[:12]
        option=dict(row,id=option_id,normal=normal.tolist(),
                    ball_center=(point+.01325*normal).tolist(),
                    usable_cameras=[cam for cam,v in row['views'].items() if v['usable']])
        options.append(option)
        render(snapshot,snapshot,option,cameras)
        if len(options)==3:
            break
    review_id=hashlib.sha256('|'.join(sorted(o['id'] for o in options)).encode()).hexdigest()[:12]
    payload=dict(experiment=report['experiment'],snapshot=str(snapshot.resolve()),review_id=review_id,
                 selected_point=report['selected_point'],options=options,
                 geometry=report.get('geometry'),
                 instructions='Label preferred / acceptable / bad / unsure, or none acceptable. Labels do not apply contacts.')
    (snapshot/'press_review.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# Press candidates for review','',
           'These candidates passed the experimental gates; that is not a manipulation-success guarantee. '
           'Choose preferred / acceptable / bad / unsure, or none acceptable. '
           'Return the snapshot and option ID with a short reason. Labels do not apply contacts.','']
    if report.get('geometry'):
        lines += ['![Pivot and pull direction](geometry.jpg)', '']
    if not options:
        lines+=['No locally supported candidates: abstain and resnapshot/reposition.','']
    for o in options:
        lines += [f'## {o["id"]}','',f'![Candidate]({o["id"]}.jpg)','',
                  f'Usable cameras: {", ".join(o["usable_cameras"])}. '
                  f'Torque / pull force: {o["torque_per_pull_N_m"]:.5f} m.','']
    (snapshot/'PRESS_REVIEW.md').write_text('\n'.join(lines))
    # Stable contact/pivot IDs prevent labels being reassigned when ranking changes.
    feedback=snapshot/'press_feedback.csv'
    rows=[]
    if feedback.exists():
        with feedback.open(newline='') as stream:
            rows=list(csv.DictReader(stream))
    existing={row['option'] for row in rows}
    for option_id in [o['id'] for o in options]+['none_acceptable_'+review_id]:
        if option_id not in existing:
            rows.append(dict(option=option_id,judgment='',reason=''))
    with feedback.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['option','judgment','reason'])
        writer.writeheader();writer.writerows(rows)
