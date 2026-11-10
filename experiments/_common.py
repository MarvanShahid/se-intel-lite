import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPOS = ["requests", "flask", "xarray", "pytest"]
OUT_ROOT = ROOT          # experiments may redirect outputs (e.g. smoke runs) by setting _common.OUT_ROOT


def _out(path) -> Path:
    p = OUT_ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _show(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def write_json(path, obj):
    p = _out(path)
    p.write_text(json.dumps(obj, indent=2, default=float))
    print(f"wrote {_show(p)}")


def write_csv(path, rows):
    import csv
    p = _out(path)
    if not rows:
        p.write_text("")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {_show(p)}")
