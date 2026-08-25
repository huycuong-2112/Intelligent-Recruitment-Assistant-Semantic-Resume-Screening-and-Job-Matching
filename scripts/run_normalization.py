"""Run deterministic Stage 4 normalization over parsed JSON files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.Normalization.cv_normalizer import normalize_cv
from src.Normalization.jd_normalizer import normalize_jd


def process(kind: str, input_dir: Path, output_dir: Path, domain: str | None) -> tuple[int, int]:
    files = sorted(input_dir.glob("*.json")) if input_dir.exists() else []
    ok = failed = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    normalizer = normalize_cv if kind == "CV" else normalize_jd
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload if isinstance(payload, list) else [payload]
            normalized = [normalizer(item, domain=domain) for item in records]
            result = normalized if isinstance(payload, list) else normalized[0]
            (output_dir / path.name).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[OK] {path.name} -> {output_dir / path.name}")
            ok += 1
        except (json.JSONDecodeError, OSError, TypeError, ValueError, AttributeError) as exc:
            print(f"[ERROR] {path}: {type(exc).__name__}: {exc}")
            failed += 1
    return ok, failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--domain", type=str, required=True, help="Explicit domain context, e.g. IT")
    args = parser.parse_args()
    for path in ("Parsed", "Normalized", "Embeddings"):
        for kind in ("CV", "JD"):
            (args.root / "Data" / path / args.domain / kind).mkdir(parents=True, exist_ok=True)
    for path in ("GroundTruth", "Results"):
        (args.root / "Data" / path / args.domain).mkdir(parents=True, exist_ok=True)
    total_ok = total_failed = 0
    for kind in ("CV", "JD"):
        ok, failed = process(kind, args.root / "Data" / "Parsed" / args.domain / kind, args.root / "Data" / "Normalized" / args.domain / kind, args.domain)
        total_ok += ok; total_failed += failed
    print(f"Summary: {total_ok} processed, {total_failed} failed")
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
