#!/usr/bin/env python3
"""CPU tests for raw/normalized adapter and actual Euler endpoint identities."""
import sys
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from utils.latent_generation_metrics import fit_statistics, normalize
from utils.sampler_contract import ConstantClean, DeterministicClean, check_endpoint
from models.single_target_generator import SingleTargetGenerator


def main():
    torch.manual_seed(21)
    c = torch.randn(1, 1, 4, 6)*3+17
    y = torch.randn_like(c)*.7-9
    cs, ys = fit_statistics(c), fit_statistics(y)
    cn, yn = normalize(c, cs), normalize(y, ys)
    # Distinct distributions deliberately catch feeding normalized anchor into
    # the historical raw-latent predictor or normalizing with the wrong stats.
    class Predictor(torch.nn.Module):
        def forward(self, anchor, target_index):
            assert target_index == 1
            return anchor*.2-13
    predictor = Predictor()
    det = DeterministicClean(predictor, cs, ys)
    expected = normalize(predictor(c[:, 0], 1)[:, None], ys)
    torch.testing.assert_close(det(y, torch.zeros(1), cn), expected)
    real_predictor = SingleTargetGenerator(latent_dim=6, num_tokens=4,
        hidden_dim=24, depth=1).eval()
    with torch.no_grad():
        real_predictor.output[-1].weight.normal_(std=.02)
        real_expected = normalize(real_predictor(c[:, 0], 1)[:, None], ys)
        real_adapter = DeterministicClean(real_predictor, cs, ys)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = real_adapter(y, torch.zeros(1), cn)
        torch.testing.assert_close(actual, real_expected)
    decode = lambda x: x.sigmoid()
    for dtype in (torch.float32, torch.bfloat16):
        for model, target in ((ConstantClean(yn), yn), (det, expected)):
            for alpha in (1., 3.):
                rows, _ = check_endpoint(model, cn, target, ys, decode,
                    [42, 99], [1, 7, 30], dtype, alpha)
                assert all(r['passed'] for r in rows), rows
    rows, _ = check_endpoint(ConstantClean(yn+1), cn, yn, ys, decode, [42], [7])
    assert not all(r['passed'] for r in rows), 'wrong clean endpoint must fail'
    try:
        check_endpoint(ConstantClean(yn), cn, yn, ys, decode, [], [7])
    except ValueError:
        pass
    else:
        raise AssertionError('empty test matrix cannot pass')
    events = []
    check_endpoint(ConstantClean(yn), cn, yn, ys, decode, [42], [1, 7], progress=events.append)
    assert len(events) == 2 and 'steps=7' in events[-1]
    import validate_r7_sampler_contract as entry
    with tempfile.TemporaryDirectory() as directory:
        def fail(args, progress):
            progress('waiting_for_video')
            raise RuntimeError('simulated read failure')
        with patch.object(entry, 'parse_args', return_value=SimpleNamespace(output_dir=directory)), \
             patch.object(entry, 'run', side_effect=fail), patch.object(entry.signal, 'signal'):
            try:
                entry.main()
            except RuntimeError:
                pass
            else:
                raise AssertionError('read failure must propagate')
        state = json.loads((Path(directory)/'contract_status.json').read_text())
        assert state['phase'] == 'waiting_for_video' and state['status'] == 'failed'
        assert state['passed'] is False and 'simulated read failure' in state['error']
    print('PASS: FP32/BF16 Euler grids, raw deterministic adapter, negative gates')
    print('PASS: per-case progress and failure records retain last phase')


if __name__ == '__main__':
    main()
