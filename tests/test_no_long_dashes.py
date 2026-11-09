from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_no_long_dashes():
    for path in ROOT.rglob("*"):
        if path.is_file() and not {".git", ".venv", "runs"} & set(path.parts) and path.suffix in {".md", ".py", ".json", ".yml", ".toml"}:
            text = path.read_text(encoding="utf-8")
            assert chr(0x2014) not in text and chr(0x2013) not in text, path
