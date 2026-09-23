"""Reject application code and run artifacts from the distributable wheel."""
import sys
from zipfile import ZipFile


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/check_wheel.py PATH.whl")
    with ZipFile(sys.argv[1]) as wheel:
        names = wheel.namelist()
        code = [name for name in names if name.endswith(".py")]
        assert code and all(name.startswith("white_matter/") for name in code), code
        forbidden = {"training", "evals", "recipes", "tests", "outputs", "checkpoints", "migration", "__pycache__"}
        assert not any(forbidden.intersection(name.split("/")) for name in names), names
        assert not any(name.endswith((".pt", ".safetensors", ".pyc", "entry_points.txt")) for name in names)
        for required in ("ops/cyclic_attention/functional.py", "blocks/lckv.py", "models/white_matter/modeling_white_matter.py"):
            assert f"white_matter/{required}" in names
        assert "white_matter/py.typed" in names, "missing PEP 561 typing marker"
        for license_file in ("LICENSE", "NOTICE", "LICENSES/Apache-2.0.txt"):
            assert any(name.endswith(f".dist-info/licenses/{license_file}") for name in names), license_file
    print(f"wheel content valid: {len(code)} Python files")


if __name__ == "__main__":
    main()
