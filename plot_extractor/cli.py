"""Batch command line: plot-extractor INPUT... -o OUT"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .config import setting


def _collect(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for i in inputs:
        p = Path(i)
        if p.is_dir():
            files += sorted(p.rglob("*.pdf")) + sorted(p.rglob("*.PDF"))
        elif p.suffix.lower() == ".pdf":
            files.append(p)
    return files


def _run_one(path: str, out: str, overlays: bool, replots: bool, llm_url: str | None, llm_model: str) -> dict:
    from .extract import extract_pdf, write_outputs
    t0 = time.time()
    try:
        res = extract_pdf(path)
        if llm_url:
            from .llm import normalize
            part = Path(path).stem
            for p in res.plots:
                p.metadata = normalize(p.to_dict(), llm_url, llm_model, part)
        write_outputs(res, out, overlays=overlays, replots=replots)
        series = [s for p in res.plots for s in p.series]
        return {"source": path, "ok": not res.error, "error": res.error,
                "plots": len(res.plots), "series": len(series),
                "low_confidence_series": sum(1 for s in series if s.confidence < 0.5),
                "raster_figures": len(res.raster_pages),
                "warnings": res.warnings,
                "seconds": round(time.time() - t0, 2)}
    except Exception as e:  # keep the batch going
        return {"source": path, "ok": False, "error": f"{type(e).__name__}: {e}",
                "seconds": round(time.time() - t0, 2)}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="plot-extractor",
                                 description="Extract plot data from vector PDF datasheets (offline).")
    ap.add_argument("inputs", nargs="+", help="PDF files and/or directories (searched recursively)")
    ap.add_argument("-o", "--out", default="out", help="output directory (default: out)")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1, help="parallel workers")
    ap.add_argument("--no-overlay", action="store_true",
                    help="skip overlay PNGs (or PLOTX_OVERLAY=false in plot-extractor.env / env)")
    ap.add_argument("--no-replot", action="store_true",
                    help="skip re-plotted PNGs (or PLOTX_REPLOT=false in plot-extractor.env / env)")
    ap.add_argument("--llm-url", default=setting("PLOTX_LLM_URL"),
                    help="OpenAI-compatible base URL for optional metadata normalization, "
                         "e.g. http://hermes.local:8080/v1 (env PLOTX_LLM_URL)")
    ap.add_argument("--llm-model", default=setting("PLOTX_LLM_MODEL", "local"),
                    help="model name to send to the LLM server (env PLOTX_LLM_MODEL)")
    a = ap.parse_args(argv)

    from .extract import env_flag
    overlays = env_flag("PLOTX_OVERLAY") and not a.no_overlay
    replots = env_flag("PLOTX_REPLOT") and not a.no_replot
    files = _collect(a.inputs)
    if not files:
        print("no PDFs found", file=sys.stderr)
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = out / "index.jsonl"
    n_ok = 0
    with index.open("a") as fh, ProcessPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        futs = {ex.submit(_run_one, str(f), str(out), overlays, replots, a.llm_url, a.llm_model): f
                for f in files}
        for k, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            n_ok += r["ok"]
            status = ("ok " if r.get("plots") else "WARN") if r["ok"] else "ERR"
            extra = (f"{r.get('plots', 0)} plots, {r.get('series', 0)} series"
                     + (f", {r['low_confidence_series']} low-conf" if r.get("low_confidence_series") else "")
                     + (f", {r['raster_figures']} raster figs skipped" if r.get("raster_figures") else "")
                     if r["ok"] else r["error"])
            print(f"[{k}/{len(files)}] {status} {Path(r['source']).name}: {extra}", flush=True)
    print(f"done: {n_ok}/{len(files)} ok -> {out}/ (index: {index})")
    return 0 if n_ok == len(files) else 1


if __name__ == "__main__":
    sys.exit(main())
