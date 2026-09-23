"""RGB perceptual loss without importing depth/video readers."""
import torch.nn.functional as F


def get_lpips(device):
    import lpips
    return lpips.LPIPS(net='vgg').to(device).eval().requires_grad_(False)


def lpips_chunked(model, pred, target, chunk_size=1, resize=256):
    pred = pred.permute(0,1,4,2,3).flatten(0,1)
    target = target.permute(0,1,4,2,3).flatten(0,1)
    if resize:
        pred=F.interpolate(pred,(resize,resize),mode='bilinear',align_corners=False)
        target=F.interpolate(target,(resize,resize),mode='bilinear',align_corners=False)
    total=pred.new_zeros(())
    for i in range(0,len(pred),chunk_size):
        total=total+model(pred[i:i+chunk_size]*2-1,target[i:i+chunk_size]*2-1).sum()
    return total/len(pred)
