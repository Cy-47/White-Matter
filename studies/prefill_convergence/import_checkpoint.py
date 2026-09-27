"""Import the trusted legacy exact-AR control without changing its tensors."""

import argparse
import json
from pathlib import Path

from benchmarks._measurement import digest, write_json
from scripts.import_paper_eval_checkpoints import convert
from studies.prefill_convergence.protocol import RECIPE
from training.recipes import load_recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(Path(str(args.checkpoint) + ".meta.json").read_text())
    expected = {
        "step": 800,
        "forward_mode": "ar",
        "student_type": "white_matter",
        "cyclic_t_n_chunks": 8,
        "seed": 1337,
        "checkpoint_format": "hf_full_state_dict_v1",
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint metadata differs from the exact-AR control")
    recipe = load_recipe(RECIPE)
    config = recipe.model
    config.training_step = metadata["step"]
    config.training_sequence_length = recipe.data.sequence_length
    config.recipe_name = recipe.name
    fingerprint = digest(args.checkpoint)
    convert(recipe.name, args.checkpoint, args.output, config=config)
    if digest(args.checkpoint) != fingerprint:
        raise RuntimeError("checkpoint changed during conversion")
    write_json(
        args.output / "import.json",
        {"source_sha256": fingerprint, "metadata": metadata, "recipe_sha256": digest(RECIPE)},
    )


if __name__ == "__main__":
    main()
