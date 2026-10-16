"""Folder-based processing: unprocessed/ -> output/ + processed/ (or failed/).

    python -m plot_extractor.inbox [--inbox unprocessed] [--processed processed]
                                   [--failed failed] [--out output]

Every PDF under the inbox (searched recursively) is extracted into
``<out>/<pdf stem>/``. On success the PDF is moved to ``processed/``; if it
can't be opened or extraction raises, it goes to ``failed/`` so it isn't
retried on every run. A run log is appended to ``<out>/index.jsonl``, and a
Markdown summary is printed (and written to $GITHUB_STEP_SUMMARY when set).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _unique(dest: Path) -> Path:
    """Avoid overwriting an earlier file with the same name."""
    if not dest.exists():
        return dest
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return dest.with_name(f"{dest.stem}_{stamp}{dest.suffix}")


def process_inbox(inbox: Path, processed: Path, failed: Path, out: Path,
                  overlays: bool = True, replots: bool = True, llm_url: str | None = None,
                  llm_model: str = "local") -> list[dict]:
    from .extract import extract_pdf, write_outputs

    pdfs = sorted(p for p in inbox.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for pdf in pdfs:
        t0 = time.time()
        rec = {"source": str(pdf.relative_to(inbox)), "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        try:
            res = extract_pdf(pdf)
            if res.error:
                raise RuntimeError(res.error)
            if llm_url:
                from .llm import normalize
                for p in res.plots:
                    p.metadata = normalize(p.to_dict(), llm_url, llm_model, pdf.stem)
            write_outputs(res, out, overlays=overlays, replots=replots)
            series = [s for p in res.plots for s in p.series]
            rec.update(ok=True, plots=len(res.plots), series=len(series),
                       low_confidence_series=sum(1 for s in series if s.confidence < 0.5),
                       raster_figures=len(res.raster_pages))
            dest_root = processed
        except Exception as e:
            rec.update(ok=False, error=f"{type(e).__name__}: {e}")
            dest_root = failed
        dest = _unique(dest_root / pdf.relative_to(inbox))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(pdf), dest)
        rec["moved_to"] = str(dest)
        rec["seconds"] = round(time.time() - t0, 2)
        results.append(rec)
        with (out / "index.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
    return results


def summary_markdown(results: list[dict]) -> str:
    if not results:
        return "No PDFs found in the inbox.\n"
    ok = sum(r["ok"] for r in results)
    lines = [f"### Datasheet extraction: {ok}/{len(results)} succeeded", "",
             "| datasheet | status | plots | series | low-confidence | raster figs skipped |",
             "|---|---|---|---|---|---|"]
    for r in results:
        if r["ok"]:
            lines.append(f"| {r['source']} | ok | {r['plots']} | {r['series']} | "
                         f"{r['low_confidence_series']} | {r['raster_figures']} |")
        else:
            lines.append(f"| {r['source']} | **failed**: {r['error']} | | | | |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inbox", default="unprocessed")
    ap.add_argument("--processed", default="processed")
    ap.add_argument("--failed", default="failed")
    ap.add_argument("--out", default="output")
    ap.add_argument("--no-overlay", action="store_true", help="skip overlay PNGs (or env PLOTX_OVERLAY=false)")
    ap.add_argument("--no-replot", action="store_true", help="skip re-plotted PNGs (or env PLOTX_REPLOT=false)")
    ap.add_argument("--llm-url", default=os.environ.get("PLOTX_LLM_URL") or None)
    ap.add_argument("--llm-model", default=os.environ.get("PLOTX_LLM_MODEL", "local"))
    a = ap.parse_args(argv)

    from .extract import env_flag
    results = process_inbox(Path(a.inbox), Path(a.processed), Path(a.failed), Path(a.out),
                            overlays=env_flag("PLOTX_OVERLAY") and not a.no_overlay,
                            replots=env_flag("PLOTX_REPLOT") and not a.no_replot,
                            llm_url=a.llm_url, llm_model=a.llm_model)
    md = summary_markdown(results)
    print(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write(md)
    # a failed datasheet is reported, not fatal: it has been moved to failed/
    return 0


if __name__ == "__main__":
    sys.exit(main())
