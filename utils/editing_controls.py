"""Optional legal source-video box controls, separate from text-only evaluation."""
import json
from pathlib import Path
import torch
from utils.editing_data import digest


def load_controls(path):
    if not path: return None
    value=json.loads(Path(path).read_text(encoding='utf8'))
    if value.get('schema')!='source-box-controls-v1' or value.get('annotation_source')!='manual_source_video':
        raise ValueError('Controls must declare source-video-only manual annotations')
    if not isinstance(value.get('cases'),dict): raise ValueError('Expected case-id keyed controls')
    for case in value['cases'].values():
        keys=case.get('keyframes',[])
        if not keys or any(len(k)!=5 for k in keys): raise ValueError('Expected [frame,x0,y0,x1,y1]')
        times=[k[0] for k in keys]
        if times!=sorted(set(times)) or any(int(t)!=t or t<0 for t in times):
            raise ValueError('Box frames must be unique ordered nonnegative integers')
        for _,x0,y0,x1,y1 in keys:
            if not (0<=x0<x1<=1 and 0<=y0<y1<=1): raise ValueError('Invalid normalized source box')
    return dict(value,identity=digest(value))


def source_box_mask(case, spatial, frames):
    """Use only an explicit annotation and source spatial metadata; no target."""
    ch,cw=spatial['canvas'];top,left,h,w=spatial['valid_rect']
    keys=case['keyframes']
    if keys[-1][0]>=frames: raise ValueError('Control frame outside source window')
    mask=torch.zeros(1,frames,ch,cw)
    for t in range(frames):
        before=max((k for k in keys if k[0]<=t),default=keys[0],key=lambda k:k[0])
        after=min((k for k in keys if k[0]>=t),default=keys[-1],key=lambda k:k[0])
        alpha=0. if before[0]==after[0] else (t-before[0])/(after[0]-before[0])
        x0,y0,x1,y1=[a+(b-a)*alpha for a,b in zip(before[1:],after[1:])]
        yy0=top+int(y0*h);yy1=top+min(h,int(y1*h+.999999))
        xx0=left+int(x0*w);xx1=left+min(w,int(x1*w+.999999))
        mask[:,t,yy0:yy1,xx0:xx1]=1
    return mask
