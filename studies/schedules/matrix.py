"""The closed 24-cell by two-seed Figure 7a recipe matrix."""

from __future__ import annotations

import re
from pathlib import Path

from studies.protocol import PAPER_SMALL_MODEL, validate_paper_training_recipe

ROOT = Path(__file__).resolve().parent / "recipes"
SEEDS = (1337, 1338)
NO_GRAD = (1, 2, 4)
GRAD = (1, 2)
MODES = ("tp", "c4", "c8", "c16")
_ARM = re.compile(r"ng([124])_g([12])_(tp|c4|c8|c16)\Z")


def evaluation_horizon(arm: str, mode: str) -> int:
    """Paper observation ceilings; native and C16 references stay at 32."""
    arm_values(arm)
    if mode not in ("native", "cyclic16", "tp"):
        raise ValueError(f"unknown evaluation mode: {mode}")
    return {"ng4_g1_c8": 128, "ng4_g2_c4": 96, "ng4_g2_c16": 96}.get(arm, 32) if mode == "tp" else 32


def arm_values(arm: str) -> tuple[int, int, str]:
    match = _ARM.fullmatch(arm)
    if match is None:
        raise ValueError(f"unknown Figure 7a schedule arm: {arm!r}")
    return int(match[1]), int(match[2]), match[3]


def recipe_path(seed: int, arm: str) -> Path:
    if seed not in SEEDS:
        raise ValueError(f"Figure 7a seed must be one of {SEEDS}")
    arm_values(arm)
    return ROOT / f"seed{seed}" / f"{arm}.yaml"


def validate_recipe(recipe, path: str | Path) -> tuple[int, str]:
    path = Path(path).resolve()
    if path.parent.parent != ROOT:
        raise ValueError("Figure 7a recipe must reside in the study recipe matrix")
    try:
        seed = int(path.parent.name.removeprefix("seed"))
        no_grad, grad, mode = arm_values(path.stem)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid Figure 7a recipe path") from exc
    if path != recipe_path(seed, path.stem) or recipe.name != f"schedules_seed{seed}_{path.stem}":
        raise ValueError("Figure 7a recipe path and name disagree")
    if (recipe.seed, recipe.steps, recipe.global_batch_size, recipe.gradient_accumulation_steps) != (
        seed,
        20_000,
        8,
        1,
    ):
        raise ValueError("Figure 7a training budget differs from the paper")
    if (recipe.no_gradient_passes, recipe.gradient_passes, recipe.model.num_passes) != (no_grad, grad, no_grad + grad):
        raise ValueError("Figure 7a pass counts differ from the arm name")
    expected_mode = "jacobi" if mode == "tp" else "cyclic"
    expected_groups = 1 if mode == "tp" else int(mode[1:])
    if (recipe.model.execution_mode, recipe.model.cyclic_groups) != (expected_mode, expected_groups):
        raise ValueError("Figure 7a execution schedule differs from the arm name")
    if recipe.model.checkpoint_jacobi_passes != (mode == "tp"):
        raise ValueError("Figure 7a Jacobi checkpoint policy differs from the paper")
    if (recipe.model.num_kv_channels, recipe.model.router_prior, recipe.model.router_layer_stride) != (
        8,
        "cyclic:0.25",
        2,
    ):
        raise ValueError("Figure 7a architecture differs from the paper")
    if recipe.data.sequence_length != 2048:
        raise ValueError("Figure 7a requires 2048-token cache rows")
    from white_matter.models.white_matter import WhiteMatterConfig

    expected_model = WhiteMatterConfig(
        **PAPER_SMALL_MODEL,
        num_kv_channels=8,
        include_top_output=False,
        use_dummy_token=True,
        num_passes=no_grad + grad,
        prefill_mode="cyclic",
        cyclic_groups=expected_groups,
        router_layer_stride=2,
        router_prior="cyclic:0.25",
        execution_mode=expected_mode,
        checkpoint_jacobi_passes=mode == "tp",
    )
    validate_paper_training_recipe(recipe, expected_model)
    return seed, path.stem
