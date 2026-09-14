"""Compare a fixed evaluation cohort against its reviewed reconstruction baseline."""
import math


def compare_replay(result, reference):
    if reference['schema'] != 'reviewed-ae-replay-v1':
        raise ValueError('unsupported AE reference')
    tolerance = float(reference['max_clip_psnr_delta'])
    if not math.isfinite(tolerance) or not 0 < tolerance <= .1:
        raise ValueError('invalid AE replay tolerance')
    for field in ('ae_signature', 'temporal_norm'):
        if result[field] != reference[field]:
            raise ValueError('AE reference mismatch: ' + field)
    def index(rows):
        values = {r['video_id']: float(r['ae_psnr_full_vs_raw']) for r in rows}
        if len(values) != len(rows) or not values or not all(math.isfinite(v) for v in values.values()):
            raise ValueError('invalid/duplicate AE reference or replay rows')
        return values
    actual, expected = index(result['clips']), index(reference['clips'])
    if actual.keys() != expected.keys():
        raise ValueError('AE replay evaluation IDs changed')
    deviations = {k: abs(actual[k]-expected[k]) for k in expected}
    return dict(passed=max(deviations.values()) <= tolerance,
                max_clip_psnr_delta=max(deviations.values()), tolerance=tolerance,
                failed_video_ids=[k for k, v in deviations.items() if v > tolerance])
