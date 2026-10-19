"""Bounded temporal-change weighting for the fixed-window x0 objective.

Change is an image-coordinate latent proxy, not foreground, optical flow or
object motion. Both camera motion and appearance changes can activate it. The
teacher supplies a detached loss weight only; it never enters model conditions.
"""
import copy
import math

import torch


def _layouts(output, clean, last_observed):
    if not all(isinstance(x, torch.Tensor) for x in (output, clean, last_observed)):
        raise TypeError('output, clean and last_observed must be tensors')
    if output.ndim != 4 or clean.shape != output.shape or min(output.shape) < 1:
        raise ValueError('output and clean must have the same nonempty [B,T,N,C] shape')
    if last_observed.shape != (clean.shape[0], 1, clean.shape[2], clean.shape[3]):
        raise ValueError('last_observed must be [B,1,N,C]')
    if output.device != clean.device or clean.device != last_observed.device:
        raise ValueError('all objective tensors must share a device')


@torch.no_grad()
def _activity(clean, last_observed, log_activity):
    clean = clean.detach().float()
    previous = torch.cat((last_observed.detach().float(), clean[:, :-1]), dim=1)
    score = (clean-previous).square().mean(dim=-1).sqrt()
    return score.log1p() if log_activity else score


@torch.no_grad()
def _error_metrics(output, clean, score, activity_floor):
    error = (output.detach().float()-clean.detach().float()).square().mean(dim=-1)
    threshold = score.mean(dim=(1, 2), keepdim=True)
    changed = (score >= threshold) & (score > activity_floor)
    changed_count = changed.sum().float()
    rest_count = (~changed).sum().float()
    global_error = error.mean()
    changed_error = torch.where(changed_count > 0,
        (error*changed).sum()/changed_count.clamp_min(1), global_error)
    rest_error = torch.where(rest_count > 0,
        (error*(~changed)).sum()/rest_count.clamp_min(1), global_error)
    names = ('x0_mse', 'changed_region_x0_mse', 'rest_region_x0_mse',
             'changed_fraction', 'changed_count', 'rest_count', 'activity_mean')
    values = torch.stack((global_error, changed_error, rest_error,
        changed.float().mean(), changed_count, rest_count, score.mean())).cpu().tolist()
    result = dict(zip(names, values))
    result['changed_count'] = int(result['changed_count'])
    result['rest_count'] = int(result['rest_count'])
    result['changed_region_fallback_global'] = result['changed_count'] == 0
    result['rest_region_fallback_global'] = result['rest_count'] == 0
    return result


def _floor(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError('activity_floor must be positive and finite')


@torch.no_grad()
def x0_error_metrics(output, clean, last_observed, *, log_activity=True,
                     activity_floor=1e-4):
    """Common unweighted teacher-forced errors, independent of loss strength.

    The changed region is score >= per-sample mean and score > activity_floor;
    empty groups report global MSE plus an explicit fallback/count. Values are
    Python scalars suitable for JSON. This helper does not run a sampler.
    """
    _layouts(output, clean, last_observed)
    _floor(activity_floor)
    score = _activity(clean, last_observed, log_activity)
    return _error_metrics(output, clean, score, activity_floor)


def motion_weighted_x0_loss(output, clean, last_observed, u, *, strength=1.,
                            cap=3., loss_floor=.05, log_activity=True,
                            activity_floor=1e-4):
    """Return (scalar loss, JSON metrics), preserving WindowFlow's time factor.

    Detached RMS change uses the last observed latent for the first future
    step and the preceding clean future latent thereafter. Optional log1p
    compresses outliers before each sample's mean normalization. Raw weights
    are clamp(1 + strength * score/mean_score, max=cap), then divided by their
    per-sample mean. Actual weights therefore have mean 1 and lie in
    [1/cap, cap]. The x0 error is divided by clamp(u, loss_floor) exactly once.

    strength=0 uses the original unweighted operation sequence exactly. This
    is target-dependent weighting, which changes the regression objective;
    it is not a claim to preserve the exact unweighted flow-matching optimum.
    """
    _layouts(output, clean, last_observed)
    _floor(activity_floor)
    if not math.isfinite(strength) or strength < 0:
        raise ValueError('strength must be nonnegative and finite')
    if not math.isfinite(cap) or cap < 1:
        raise ValueError('cap must be finite and at least 1')
    if not math.isfinite(loss_floor) or not 0 < loss_floor <= 1:
        raise ValueError('loss_floor must be in (0,1]')
    if not isinstance(u, torch.Tensor) or u.shape != (output.shape[0],) or u.device != output.device:
        raise ValueError('u must have shape [B] on the output device')
    score = _activity(clean, last_observed, log_activity)
    with torch.no_grad():
        if strength == 0:
            weight = torch.ones_like(score)
        else:
            ratio = score/score.mean(dim=(1, 2), keepdim=True).clamp_min(activity_floor)
            weight = (1 + strength*ratio).clamp(max=cap)
            weight = weight/weight.mean(dim=(1, 2), keepdim=True)
    scale = u.detach().float().clamp_min(loss_floor).view(-1, 1, 1, 1)
    squared = ((output.float()-clean.detach().float())/scale).square()
    loss = squared.mean() if strength == 0 else (squared*weight.unsqueeze(-1)).mean()
    metrics = _error_metrics(output, clean, score, activity_floor)
    weight_values = torch.stack((weight.min(), weight.max(), weight.mean(),
        (weight.mean(dim=(1, 2))-1).abs().max())).detach().cpu().tolist()
    metrics.update(zip(('weight_min', 'weight_max', 'weight_mean',
                        'weight_sample_mean_max_error'), weight_values))
    return loss, metrics


class PlateauTracker:
    """Patience on common global AND changed-region validation MSE.

    Either metric improving by more than relative_tol resets patience. Small
    improvements can accumulate against each metric's last significant best;
    ``best`` separately retains its actual lowest observed value. Patience is
    counted per distinct valid evaluation step, and min_steps is a lower bound
    on stopping, not a quality assertion. Missing/nonfinite probes are skipped.
    """
    METRICS = ('x0_mse', 'changed_region_x0_mse')

    def __init__(self, min_steps=4000, patience=4, relative_tol=.002):
        if type(min_steps) is not int or min_steps < 0:
            raise ValueError('min_steps must be a nonnegative integer')
        if type(patience) is not int or patience < 1:
            raise ValueError('patience must be a positive integer')
        if not math.isfinite(relative_tol) or not 0 <= relative_tol < 1:
            raise ValueError('relative_tol must be in [0,1)')
        self.min_steps, self.patience, self.relative_tol = min_steps, patience, float(relative_tol)
        self.best = dict.fromkeys(self.METRICS)
        self.reference_best = dict.fromkeys(self.METRICS)
        self.stale = self.updates = 0
        self.last_step = None

    @property
    def converged(self):
        return bool(self.last_step is not None and self.last_step >= self.min_steps
                    and self.stale >= self.patience)

    def _result(self, updated, reason, improved=()):
        return dict(updated=updated, reason=reason, converged=self.converged,
                    last_step=self.last_step, stale=self.stale, updates=self.updates,
                    best=dict(self.best), improved=list(improved))

    def update(self, step, metrics):
        if type(step) is not int or step < 0:
            raise ValueError('evaluation step must be a nonnegative integer')
        if self.last_step is not None and step <= self.last_step:
            return self._result(False, 'repeated_or_older_step')
        try:
            values = {key: float(metrics[key]) for key in self.METRICS}
        except (KeyError, TypeError, ValueError, OverflowError):
            return self._result(False, 'missing_or_invalid_metrics')
        if any(not math.isfinite(x) or x < 0 for x in values.values()):
            return self._result(False, 'nonfinite_or_negative_metrics')
        improved = []
        for key, value in values.items():
            reference = self.reference_best[key]
            if reference is None or reference-value > self.relative_tol*max(abs(reference), 1e-12):
                improved.append(key)
                self.reference_best[key] = value
            self.best[key] = value if self.best[key] is None else min(self.best[key], value)
        self.stale = 0 if improved else self.stale+1
        self.last_step, self.updates = step, self.updates+1
        return self._result(True, 'improved' if improved else 'plateau', improved)

    def state_dict(self):
        return dict(schema='scene-dual-metric-plateau-v1',
            config=dict(min_steps=self.min_steps, patience=self.patience, relative_tol=self.relative_tol),
            best=dict(self.best), reference_best=dict(self.reference_best), stale=self.stale,
            updates=self.updates, last_step=self.last_step, converged=self.converged)

    def load_state_dict(self, state):
        if state.get('schema') != 'scene-dual-metric-plateau-v1' or state.get('config') != self.state_dict()['config']:
            raise ValueError('Plateau state belongs to another schema or policy')
        for field in ('best', 'reference_best'):
            values = state.get(field)
            if not isinstance(values, dict) or set(values) != set(self.METRICS):
                raise ValueError('Invalid plateau metric state')
            if any(v is not None and (not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0)
                   for v in values.values()):
                raise ValueError('Invalid plateau metric value')
        if any(type(state.get(k)) is not int or state[k] < 0 for k in ('stale', 'updates')):
            raise ValueError('Invalid plateau counters')
        step = state.get('last_step')
        if step is not None and (type(step) is not int or step < 0):
            raise ValueError('Invalid plateau step')
        restored = copy.deepcopy(state)
        self.best, self.reference_best = restored['best'], restored['reference_best']
        self.stale, self.updates, self.last_step = state['stale'], state['updates'], step
