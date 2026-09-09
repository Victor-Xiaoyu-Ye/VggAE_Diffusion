"""Detached noise-bin telemetry. Never changes RNG, gradients or flow weighting."""
import torch


class NoiseMeter:
    def __init__(self, device):
        # count, u, weight, clean MSE, weighted objective, normalized target power
        self.sums = torch.zeros(5, 6, device=device, dtype=torch.float32)
        self.moments = {}

    @torch.no_grad()
    def observe_latents(self, name, x):
        # Per temporal slot/channel, reduce only batch and spatial dimensions.
        x = x.detach().float()
        value = torch.stack((x.sum((0, 2)), x.square().sum((0, 2)),
                             torch.full_like(x[0, :, 0], x.shape[0]*x.shape[2])))
        if name not in self.moments: self.moments[name] = torch.zeros_like(value)
        self.moments[name].add_(value)

    @torch.no_grad()
    def update(self, flow, output, clean, noise, u):
        noisy = (1-u[:, None, None, None])*clean+u[:, None, None, None]*noise
        error = (flow.clean(output.detach(), noisy, u)-clean.float()).square().flatten(1).mean(1)
        weight = u.float().clamp_min(flow.loss_floor).pow(-2)
        if flow.prediction == 'velocity':
            objective = (output.detach().float()-(noise.float()-clean.float())).square().flatten(1).mean(1)
            weight = torch.ones_like(u)
        else:
            objective = weight*error
        values = torch.stack((torch.ones_like(u), u, weight, error, objective,
                              clean.float().square().flatten(1).mean(1)), 1)
        self.sums.index_add_(0, (u*5).long().clamp(0, 4), values)

    def flush(self, distributed=False):
        values = self.sums.clone()
        self.sums.zero_()
        if distributed:
            torch.distributed.all_reduce(values)
        values = values.cpu().tolist()
        result = {}
        total = sum(v[0] for v in values)
        loss_total = sum(v[4] for v in values)
        for index, v in enumerate(values):
            prefix = f'train/noise_bin{index}'
            result[prefix+'/count'] = v[0]
            result[prefix+'/sample_fraction'] = v[0]/max(total, 1)
            result[prefix+'/loss_fraction'] = v[4]/max(loss_total, 1e-12)
            if v[0]:
                for name, number in zip(('u', 'weight', 'x0_mse', 'objective', 'target_power'), v[1:]):
                    result[prefix+'/'+name] = number/v[0]
        for name, accumulator in self.moments.items():
            moment = accumulator.clone(); accumulator.zero_()
            if distributed: torch.distributed.all_reduce(moment)
            moment = moment.cpu()
            mean = moment[0]/moment[2].clamp_min(1)
            std = (moment[1]/moment[2].clamp_min(1)-mean.square()).clamp_min(0).sqrt()
            prefix = 'train/normalized_'+name
            result[prefix+'/mean_abs_average'] = float(mean.abs().mean())
            result[prefix+'/mean_abs_max'] = float(mean.abs().max())
            for suffix, value in [('std_min',std.min()),('std_median',std.median()),('std_max',std.max())]:
                result[prefix+'/'+suffix] = float(value)
        return result


def normalization_report(stats):
    report = {'schema': 'window-normalization-audit-v1',
              'note': 'Cached normalization metadata; does not independently prove normalized data have unit covariance.'}
    for name in ('cond', 'target'):
        group = stats[name]
        report[name] = {}
        for key in ('mean', 'std'):
            x = group[key].detach().cpu().float()
            report[name][key] = dict(shape=list(x.shape), values=x.tolist(),
                min=float(x.min()), median=float(x.median()), max=float(x.max()))
    return report
