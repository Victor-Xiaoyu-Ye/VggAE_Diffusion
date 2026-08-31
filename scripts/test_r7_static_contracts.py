#!/usr/bin/env python3
"""Static AST checks for the R7 scale wrappers and shared imports."""
from __future__ import annotations

import ast
import os
import re


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def argument_dests(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    result = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        value = node.args[0]
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        flag = value.value
        if flag.startswith("--"):
            result.add(flag[2:].replace("-", "_"))
            for keyword in node.keywords:
                if keyword.arg == "dest" and isinstance(keyword.value, ast.Constant):
                    result.discard(flag[2:].replace("-", "_"))
                    result.add(str(keyword.value.value))
    return result


def shell_flags(path):
    text = open(path, encoding="utf-8").read()
    return {value.replace("-", "_") for value in re.findall(r"--([a-z][a-z0-9_-]*)", text)}


def defined_names(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    return {node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def imported_names(path, module):
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.update(alias.name for alias in node.names)
    return names


def main():
    wan_trainer = os.path.join(ROOT, "train_causal_wan_video_diffusion.py")
    compact_trainer = os.path.join(ROOT, "train_causal_video_diffusion.py")
    for shell, python in (
        ("scripts/scale/13_train_causal_video_diffusion.sh", compact_trainer),
        ("scripts/scale/16_train_wan_t2v_diffusion.sh", wan_trainer),
    ):
        shell_path = os.path.join(ROOT, shell)
        missing = shell_flags(shell_path) - argument_dests(python)
        # MODE=sample calls a sibling sampler, so sampler-only flags are allowed.
        allowed = {"checkpoint", "weights", "sample_index", "stats", "output",
                   "preview_fps", "decoder_ckpt", "no_enforce_quality_guards"}
        missing -= allowed
        assert not missing, f"{shell}: unknown trainer flags {sorted(missing)}"

    shared = imported_names(wan_trainer, "train_causal_video_diffusion")
    missing = shared - defined_names(compact_trainer) - {
        "CONTEXT_CHUNKS", "FUTURE_CHUNKS", "LATENT_DIM"}
    assert not missing, f"Wan trainer imports missing shared names: {sorted(missing)}"
    print("R7 static contracts passed")


if __name__ == "__main__":
    main()
