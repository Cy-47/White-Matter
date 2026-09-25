"""Optional Hugging Face integrations. Import a family or register all explicitly."""


def register_models():
    from importlib import import_module

    for family in ("white_matter", "lckv", "vanilla", "fusedkv", "feedback_transformer"):
        import_module(f"white_matter.models.{family}")
