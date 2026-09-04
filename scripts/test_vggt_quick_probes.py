#!/usr/bin/env python3
"""Static and optional tensor checks for the VGGT quick probes."""
from __future__ import annotations

import ast
import os
import re
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse(path):
    return ast.parse(open(path, encoding="utf-8").read(), filename=path)


def argparse_dests(path):
    result = set()
    required = set()
    for node in ast.walk(parse(path)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and node.args[0].value.startswith("--")):
            name = node.args[0].value[2:].replace("-", "_")
            result.add(name)
            for keyword in node.keywords:
                if (keyword.arg == "required"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True):
                    required.add(name)
    return result, required


def branch_flags(source, marker, end_marker):
    branch = source.split(marker, 1)[1].split(end_marker, 1)[0]
    return {value.replace("-", "_") for value in re.findall(
        r"--([a-z][a-z0-9_-]*)", branch)}


def shell_flags(path):
    return {value.replace("-", "_") for value in re.findall(
        r"--([a-z][a-z0-9_-]*)", open(path, encoding="utf-8").read())}


def static_checks():
    wrapper_path = os.path.join(ROOT, "scripts/scale/22_probe_vggt_generation.sh")
    wrapper = open(wrapper_path, encoding="utf-8").read()
    manifold_flags = branch_flags(
        wrapper, 'if [[ "${MODE}" == "manifold" ]]', "else")
    train_flags = branch_flags(wrapper, "else", "fi\n\n")
    manifold_dests, manifold_required = argparse_dests(
        os.path.join(ROOT, "probe_vggt_manifold.py"))
    train_dests, train_required = argparse_dests(
        os.path.join(ROOT, "train_single_target_probe.py"))
    assert not manifold_flags - manifold_dests, \
        f"manifold wrapper has unknown flags: {sorted(manifold_flags - manifold_dests)}"
    assert not train_flags - train_dests, \
        f"train wrapper has unknown flags: {sorted(train_flags - train_dests)}"
    # All required Python arguments must occur in the corresponding branch.
    assert manifold_required <= manifold_flags, \
        f"manifold wrapper misses required flags: {sorted(manifold_required - manifold_flags)}"
    assert train_required <= train_flags, \
        f"train wrapper misses required flags: {sorted(train_required - train_flags)}"
    for file in ("probe_vggt_manifold.py", "train_single_target_probe.py",
                 "check_single_target_probe.py",
                 "models/single_target_generator.py"):
        parse(os.path.join(ROOT, file))
    assert shell_flags(wrapper_path), "quick wrapper has no CLI flags"


def tensor_checks():
    try:
        import torch
    except ImportError:
        print("torch unavailable: tensor checks skipped")
        return
    from models.single_target_generator import (
        SingleTargetFlowGenerator, SingleTargetGenerator,
        euclidean_flow_sample)
    from probe_vggt_manifold import centered_unit, exp_map, slerp, tangent_project

    x = torch.randn(2, 5, 8)
    unit = centered_unit(x)
    assert torch.allclose(unit.mean(-1), torch.zeros_like(unit.mean(-1)), atol=1e-6)
    assert torch.allclose(unit.norm(dim=-1), torch.ones_like(unit[..., 0]), atol=1e-5)
    tangent = tangent_project(unit, torch.randn_like(unit))
    assert ((tangent * unit).sum(-1).abs() < 1e-5).all()
    moved = exp_map(unit, tangent * 0.01)
    assert torch.allclose(moved.norm(dim=-1), torch.ones_like(moved[..., 0]), atol=1e-4)
    middle = slerp(x, x, 0.5)
    assert torch.isfinite(middle).all()

    anchor = torch.randn(2, 16, 12)
    deterministic = SingleTargetGenerator(
        latent_dim=12, num_tokens=16, hidden_dim=48, depth=1)
    flow = SingleTargetFlowGenerator(
        latent_dim=12, num_tokens=16, hidden_dim=48, depth=1)
    assert deterministic(anchor, 1).shape == anchor.shape
    assert flow(torch.randn_like(anchor), anchor, 1, torch.rand(2)).shape == anchor.shape
    sampled = euclidean_flow_sample(
        flow, anchor, 1, anchor.shape, steps=2, seed=7)
    assert sampled.shape == anchor.shape and torch.isfinite(sampled).all()
    try:
        flow(torch.randn_like(anchor), anchor, 9, torch.rand(2))
    except ValueError:
        pass
    else:
        raise AssertionError("flow target-index validation is missing")


def main():
    static_checks()
    tensor_checks()
    print("VGGT quick-probe contracts passed")


if __name__ == "__main__":
    main()
