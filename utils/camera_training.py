"""Exact-frame camera joins and matched training subsets for the pose pilot."""
from pathlib import Path
import hashlib
import torch
from torch.utils.data import IterableDataset


def camera_key(sample):
    frames=sample.get('frame_indices')
    if frames is None or len(frames)<2:
        raise ValueError('camera conditioning requires cached exact RGB frame_indices')
    return str(sample['video_id'])+':'+','.join(str(int(x)) for x in frames)


class CameraBank:
    def __init__(self,path,frames):
        self.signature=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        bank=torch.load(path,map_location='cpu',weights_only=False)
        if bank['schema']!='r7-camera-pilot-bank-v1':raise ValueError('unsupported camera bank')
        self.features=bank['features'];self.train_keys=set(bank['train_keys']);self.eval_keys=set(bank['eval_keys'])
        if not self.train_keys or not self.eval_keys or self.train_keys & self.eval_keys:
            raise ValueError('invalid camera split keys')
        if self.train_keys|self.eval_keys != set(self.features):raise ValueError('camera feature coverage mismatch')
        for value in self.features.values():
            if value.shape!=(frames,14) or not torch.isfinite(value).all():raise ValueError('invalid camera feature')

    def batch(self,batch,device,mode='pose',dropout=0.):
        keys=[camera_key(dict(video_id=vid,frame_indices=frames))
              for vid,frames in zip(batch['video_id'],batch['frame_indices'])]
        values=torch.stack([self.features[key] for key in keys]).to(device).float()
        # Both matched arms consume the same RNG draw, even for the null arm.
        present=torch.rand(len(keys),device=device)>=dropout if dropout else torch.ones(len(keys),device=device,dtype=torch.bool)
        if mode=='null':present.zero_()
        elif mode=='wrong':
            # Fixed cyclic donor mapping of heldout paths; keep the recipient K
            # and timestamps, changing only pose. B=1 must not shuffle into itself.
            donors=sorted(self.eval_keys)
            if len(donors)<2:raise ValueError('wrong-camera evaluation requires at least two heldout paths')
            for i,key in enumerate(keys):
                donor=donors[(donors.index(key)+1)%len(donors)]
                values[i,:,:9]=self.features[donor][:,:9].to(device)
        elif mode!='pose':raise ValueError('unknown camera mode')
        return values,present


class CameraFilteredDataset(IterableDataset):
    def __init__(self,base,allowed):
        super().__init__();self.base=base;self.allowed=allowed

    def __iter__(self):
        # Base sharding remains unchanged, but rejection is identical in both
        # arms. Fail instead of spinning indefinitely on an empty rank.
        rejected=0
        for sample in self.base:
            if camera_key(sample) in self.allowed:
                rejected=0;yield sample
            else:
                rejected+=1
                if rejected>10000:raise RuntimeError('camera subset has no usable samples for this rank')


class BoundCameraModel:
    def __init__(self,model,camera,present):
        self.model,self.camera,self.present=model,camera,present

    def __call__(self,x,u,anchor,text=None,text_valid=None,**kwargs):
        return self.model(x,u,anchor,text,text_valid,camera=self.camera,camera_present=self.present,**kwargs)


def initialize_from_checkpoint(core,path,step,weights,model_args,statistics,runtime,large=False):
    saved=(torch.load(path,map_location='cpu',weights_only=False,mmap=True) if large else
           torch.load(path,map_location='cpu',weights_only=False))
    if saved.get('schema')!='r7-window-trainer-v1' or saved.get('step')!=step:
        raise ValueError('initialization checkpoint identity/step mismatch')
    expected={k:v for k,v in model_args.items() if k!='camera_dim'}
    if saved['model_args']!=expected:raise ValueError('initialization architecture mismatch')
    from utils.window_training import digest
    if (digest(saved['statistics'])!=digest(statistics) or
            saved['contract']['identity']['representation'].get('window_codec_runtime')!=runtime):
        raise ValueError('initialization representation/statistics/runtime mismatch')
    result=core.load_state_dict(saved[weights],strict=False)
    allowed={name for name in core.state_dict() if name.startswith('camera_conditioner.')}
    if set(result.missing_keys)!=allowed or result.unexpected_keys:
        raise ValueError('only new camera conditioner parameters may be absent at initialization')
    return dict(step=step,weights=weights,parent_contract_sha256=digest(saved['contract']))
