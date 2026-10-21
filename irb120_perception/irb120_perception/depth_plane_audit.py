"""Audit table and object-top planes from a snapshot's exact aligned depth frame."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def fit_plane(points, minimum=30):
    """Robustly fit z=ax+by+c and report its upward normal and residuals."""
    q=np.asarray(points,dtype=float)
    q=q[np.isfinite(q).all(axis=1)]
    if len(q)<minimum:
        return dict(available=False,reason='insufficient points',count=len(q))
    for _ in range(6):
        A=np.c_[q[:,:2],np.ones(len(q))]
        coef=np.linalg.lstsq(A,q[:,2],rcond=None)[0]
        residual=q[:,2]-A@coef
        median=np.median(residual)
        mad=np.median(np.abs(residual-median))+1e-12
        keep=np.abs(residual-median)<max(.003,3*1.4826*mad)
        if keep.all() or keep.sum()<minimum:
            break
        q=q[keep]
    A=np.c_[q[:,:2],np.ones(len(q))]
    coef=np.linalg.lstsq(A,q[:,2],rcond=None)[0]
    residual=q[:,2]-A@coef
    normal=np.array([-coef[0],-coef[1],1.])
    normal/=np.linalg.norm(normal)
    return dict(available=True,count=len(q),normal=normal.tolist(),
                tilt_deg=float(np.degrees(np.arccos(np.clip(normal[2],-1,1)))),
                rms_mm=float(np.sqrt(np.mean(residual**2))*1000),
                p95_abs_mm=float(np.quantile(np.abs(residual),.95)*1000),
                equation_z_ax_by_c=coef.tolist())


def depth_to_base(depth_mm, info, stride=2):
    depth=np.asarray(depth_mm)[::stride,::stride].astype(float)/1000
    K=np.asarray(info['K']); tf=info['base_to_optical']
    R,t=np.asarray(tf['R']),np.asarray(tf['t'])
    v,u=np.indices(depth.shape)
    u=u*stride;v=v*stride
    optical=np.stack(((u-K[0,2])*depth/K[0,0],
                      (v-K[1,2])*depth/K[1,1],depth),axis=-1)
    base=(optical-t)@R
    return base[(depth>0)&np.isfinite(base).all(axis=-1)]


def audit(snapshot, camera='cam2', table_band=(-.08, .04)):
    """table_band: base_link z range the table points are taken from."""
    snapshot=Path(snapshot)
    saved=np.load(snapshot/f'{camera}_aligned_depth.npz')
    info=json.loads((snapshot/f'{camera}_camera_info.json').read_text())
    obj=np.load(snapshot/'object_cloud.npz')['xyz']
    obj=obj[np.isfinite(obj).all(axis=1)]
    points=depth_to_base(saved['depth_mm'],info)
    low=np.quantile(obj[:,:2],[.02],axis=0)[0]
    high=np.quantile(obj[:,:2],[.98],axis=0)[0]
    pad=.02
    object_xy=((points[:,0]>=low[0]-pad)&(points[:,0]<=high[0]+pad)
               &(points[:,1]>=low[1]-pad)&(points[:,1]<=high[1]+pad))
    top_reference=float(np.quantile(obj[:,2],.995))
    span=high-low
    interior_margin=min(.015,.2*float(np.min(span)))
    try:
        hull=ConvexHull(obj[:,:2])
        signed=points[:,:2]@hull.equations[:,:2].T+hull.equations[:,2]
        interior=np.max(signed,axis=1)<=-interior_margin
    except (ValueError,QhullError):
        interior=((points[:,0]>=low[0]+interior_margin)
                  &(points[:,0]<=high[0]-interior_margin)
                  &(points[:,1]>=low[1]+interior_margin)
                  &(points[:,1]<=high[1]-interior_margin))
    top=points[interior&(points[:,2]>=top_reference-.025)
               &(points[:,2]<=obj[:,2].max()+.025)]
    near=((points[:,0]>=low[0]-.30)&(points[:,0]<=high[0]+.30)
          &(points[:,1]>=low[1]-.30)&(points[:,1]<=high[1]+.30))
    table=points[near&~object_xy&(points[:,2]>=table_band[0])&(points[:,2]<=table_band[1])]
    result=dict(snapshot=str(snapshot.resolve()),camera=camera,
                depth_stamp=float(saved['stamp']),depth_shape=list(saved['depth_mm'].shape),
                table=fit_plane(table),object_top=fit_plane(top),
                object_bounds_xy=[low.tolist(),high.tolist()],
                top_reference=top_reference,
                top_interior_margin_mm=interior_margin*1000)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot',type=Path)
    parser.add_argument('--camera',default='cam2',choices=['cam1','cam2','cam3'])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=audit(args.snapshot,args.camera)
    text=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.write_text(text)
    print(text,end='')


if __name__=='__main__':
    main()
