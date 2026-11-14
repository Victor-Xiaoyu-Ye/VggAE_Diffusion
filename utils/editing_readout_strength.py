"""Stage72 protocol and source-only strength diagnostics; no new editing operator.

The subclass observes the Stage70 writeback without modifying it. Equal alpha
is not equal dose. Matched-strength interpolation is a descriptive diagnostic,
not a generated sample, a selected operating point, or a causal guarantee.
"""
import math
import numpy as np
import torch

from utils.editing_attribution import NativeEffectFeedback, OPERATOR_ARMS, supported_rms
from utils.editing_paper_metrics import aggregate


def specifications(cfg):
    specs = {arm: dict(operator=arm, alpha=cfg['feedback']['writeback_alpha'], subset_only=False)
             for arm in OPERATOR_ARMS}
    for alpha in cfg['extra_alphas']:
        for arm in OPERATOR_ARMS[1:]:
            specs[f'{arm}_a{round(alpha*100):03d}'] = dict(operator=arm, alpha=alpha, subset_only=True)
    return specs


def validate_config(cfg):
    if cfg['schema'] != 'editing-readout-strength-v1' or (cfg['frames'], cfg['fps']) != (17, 8):
        raise ValueError('Stage72 requires the frozen 17-frame/8fps protocol')
    if cfg['extra_alphas'] != [.15, .55] or cfg['feedback']['writeback_alpha'] != .35:
        raise ValueError('Predeclared alpha grid is .15/.35/.55; do not select on test outcomes')
    if cfg['sample_steps'] < 2 or cfg['seeds'] != [42]:
        raise ValueError('Keep the Stage70 seed and a valid sampling schedule')


def select_cohorts(cohort, cfg):
    rows = cohort['records']; ids = [r['id'] for r in rows]; subset = cohort['ablation_ids']
    if (len(ids) != cfg['expected_sources'] or len(subset) != cfg['expected_subset'] or
            len(set(ids)) != len(ids) or len(set(subset)) != len(subset) or not set(subset) <= set(ids)):
        raise ValueError('Fixed parent cohort is incomplete or has duplicate IDs')
    groups = [r['group'] for r in rows]
    if len(set(groups)) != len(groups):
        raise ValueError('One row per frozen source group is required')
    selected = [r for r in rows if r['id'] in subset]
    extension = [r for r in rows if r['id'] not in subset]
    if cfg.get('smoke'):
        selected, extension = selected[:3], extension[:3]
    return selected, extension


def records_for(m, cohort='all', arm=None):
    ids = set(m['subset_ids'])
    if arm and specifications(m['config'])[arm]['subset_only']:
        if cohort == 'extension': return []
        cohort = 'subset'
    return [r for r in m['records'] if cohort == 'all' or (r['id'] in ids) == (cohort == 'subset')]


def arms_for(m, row):
    return [arm for arm, spec in specifications(m['config']).items()
            if not spec['subset_only'] or row['id'] in m['subset_ids']]


class ObservedEffectFeedback(NativeEffectFeedback):
    """Keep the numerical update byte-for-byte; log its actual post-cap dose."""
    @torch.no_grad()
    def __call__(self, *, step_index, sigma, latent, velocity):
        observer = self.observer
        self.observer = None
        before = len(self.events)
        try:
            result = super().__call__(step_index=step_index, sigma=sigma, latent=latent, velocity=velocity)
        finally:
            self.observer = observer
        if len(self.events) > before:
            event = self.events[-1]
            if event['applied']:
                delta = (velocity.float()-result.float())*float(sigma)
                event.update(alpha=float(self.cfg['writeback_alpha']),
                    reference_prefix_delta_rms=float(delta[:, :1].square().mean().sqrt()),
                    # Slot 0 is the reference prefix; slot 1 is the first video frame.
                    clean_delta_future_rms=float(delta[:, 2:].square().mean().sqrt()),
                    velocity_delta_rms=float((delta[:, 1:]/float(sigma)).square().mean().sqrt()),
                    dose_scope='post-cap native latent; future excludes reference prefix and first video slot')
            if observer: observer(event)
        return result


def strength_metrics(generated, reference):
    """Final-frame effects on SHARED source-derived support, excluding anchor.

    Projection onto the common proposed edit distinguishes editing from merely
    changing pixels. It is a reference-agreement proxy, not instruction accuracy.
    """
    y,x,h,w = reference['spatial']['valid_rect']
    crop = lambda a: a.detach().cpu().float()[1:, :, y:y+h, x:x+w]
    effect = crop(generated)-crop(reference['source'])
    wanted = crop(reference['reference'])-crop(reference['source'])
    mask = (crop(reference['masks'])*crop(reference['confidence'])).clamp(0, 1)
    valid = float(mask.sum()) > 1e-8
    denom = float((wanted.square()*mask).sum())
    return dict(future_edit_rms=float(supported_rms(effect, mask)) if valid else None,
        future_edit_l1=float((effect.abs()*mask).sum()/(3*mask.sum())) if valid else None,
        future_reference_projection=float((effect*wanted*mask).sum())/denom if denom > 1e-10 else None,
        future_reference_error_rms=float(supported_rms(effect-wanted, mask)) if valid else None,
        future_outside_support_rms=float(supported_rms(effect, 1-crop(reference['masks']).clamp(0,1))),
        future_support_fraction=float(mask.mean()), future_all_pixel_rms=float(effect.square().mean().sqrt()))


def _curve(points, key):
    if len(points) != 3: return None, 'missing_alpha'
    ordered = sorted(points, key=lambda p:p['alpha'])
    x = [p['metrics'].get(key) for p in ordered]
    if any(v is None or not math.isfinite(v) for v in x): return None, 'undefined_strength'
    d = np.diff(x)
    if not (np.all(d > 1e-8) or np.all(d < -1e-8)): return None, 'flat_or_nonmonotonic_strength'
    return sorted(ordered, key=lambda p:p['metrics'][key]), None


def matched_strength(records, cfg, expected=None):
    """Per-source overlap midpoint fixed by strength ONLY, no extrapolation.

    Return exclusions and all curves. No quality-based selection, no best alpha,
    no declaration that interpolation constitutes a measured operating point.
    """
    specs = specifications(cfg)
    points = {(r['id'],r['group'],seed):{} for r in (expected or []) for seed in cfg['seeds']}
    for r in records:
        if r.get('status') != 'completed' or r['arm'] == 'op_sft': continue
        spec = specs[r['arm']]
        points.setdefault((r['id'],r['group'],r['seed_base']),{}).setdefault(spec['operator'],[]).append(
            dict(r,alpha=spec['alpha']))
    reports = {}
    for strength in ('future_edit_rms','future_reference_projection'):
        controls = {}
        for control in OPERATOR_ARMS[1:-1]:
            pairs=[]; excluded=[]; synthetic=[]
            for (key,group,seed), arms in points.items():
                ca,reason_a = _curve(arms.get('op_rae',[]),strength)
                cb,reason_b = _curve(arms.get(control,[]),strength)
                if reason_a or reason_b:
                    excluded.append(dict(id=key,seed_base=seed,reason=reason_a or reason_b));continue
                xa=[p['metrics'][strength] for p in ca];xb=[p['metrics'][strength] for p in cb]
                lo=max(xa[0],xb[0]);hi=min(xa[-1],xb[-1])
                if hi-lo < 1e-8:
                    excluded.append(dict(id=key,seed_base=seed,reason='no_common_strength_interval'));continue
                query=(lo+hi)/2; values={}
                for name,curve,xx in [('rae_full',ca,xa),(control,cb,xb)]:
                    metrics={}
                    for metric in curve[0]['metrics']:
                        yy=[p['metrics'].get(metric) for p in curve]
                        if all(isinstance(y,(float,int)) and math.isfinite(y) for y in yy):
                            metrics[metric]=float(np.interp(query,xx,yy))
                    values[name]=metrics
                    synthetic.append(dict(id=key,group=group,seed_base=seed,arm=name,status='completed',metrics=metrics))
                pairs.append(dict(id=key,seed_base=seed,query=query,overlap=[lo,hi],values=values))
            controls[control]=dict(eligible=len(points),matched=len(pairs),excluded=excluded,points=pairs,
                statistics=aggregate(synthetic,['rae_full',control],bootstrap=10000))
        reports[strength]=controls
    return dict(scope='Diagnostic interpolation, not newly generated or tuned outputs; report exclusions with paired statistics',
        selection='per-source midpoint of common strength range; reject nonmonotonic curves; never extrapolate',comparisons=reports)
