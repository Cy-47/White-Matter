"""Protocol and artifact checks for the main-body prefill convergence experiment."""

import math
from pathlib import Path

from training.recipes import load_recipe, model_recipe_keys

GROUPS = (2, 4, 8, 16, 32, 64)
MODES = {"jacobi": None, **{f"{kind}{g}": g for kind in ("cyclic", "contiguous") for g in GROUPS}}
RECIPE = Path(__file__).parent / "recipes/exact_ar_4l.yaml"


def schedules(modes):
    return {mode: dict(partition="contiguous" if mode.startswith("contiguous") else
                       "cyclic" if mode.startswith("cyclic") else "jacobi",
                       groups=MODES[mode], update="after_group", version=1) for mode in modes}


def validate_checkpoint(config):
    recipe = load_recipe(RECIPE)
    for key in model_recipe_keys(type(recipe.model)):
        if getattr(config, key, None) != getattr(recipe.model, key, None):
            raise ValueError(f"convergence checkpoint differs from recipe: {key}")
    for key, value in {"training_step": 800, "training_sequence_length": 1024,
                       "recipe_name": recipe.name}.items():
        if getattr(config, key, None) != value:
            raise ValueError(f"convergence checkpoint {key} must be {value}")


def first_crossing(losses, reference, tolerance=0.01):
    if not math.isfinite(reference) or not 0 <= tolerance < 1:
        raise ValueError("invalid convergence reference or tolerance")
    if any(not math.isfinite(value) for value in losses):
        raise ValueError("nonfinite convergence curve")
    target = reference + math.log1p(tolerance)
    return next((i for i, value in enumerate(losses, 1) if value <= target), None)
