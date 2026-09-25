"""Evaluate a shared-mixture HF export with the paper downstream suite."""

from studies.shared_mixture.model import register_model


def main() -> None:
    register_model()
    from evals.lm_eval import main as evaluate_main

    evaluate_main()


if __name__ == "__main__":
    main()
