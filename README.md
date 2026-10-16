# plot-extractor

Offline extraction of curve data from plots in PDF datasheets, built for batch runs over large datasheet libraries.

Most modern datasheets draw their plots as vector graphics. This tool reads the PDF drawing commands directly instead of looking at pixels, so extracted points sit on the exact coordinates the plotting software wrote. No network access is needed. An optional local LLM can normalize metadata, but it never produces the numbers.

```
pip install -e .
plot-extractor datasheets/ -o out/ -j 8
```

For each PDF this writes, per plot:

| file | contents |
|---|---|
| `*.json` | title, axis labels and scale (lin/log), tick values, calibration residual, and each series with label, confidence, notes and `[x, y]` points |
| `*.csv` | `series, x, y` |
| `*_overlay.png` | the extracted points drawn over a render of the original plot, for spot checks |
| `summary.json` | per-document roll-up, including figures that could not be processed |

`out/index.jsonl` gets one line per PDF (status, plot and series counts, number of low-confidence series, time taken). Sort or filter it to triage a batch.

## How it works

1. **Primitives** (`primitives.py`) normalize every way a curve can be encoded into centerline segments:
   - native stroked paths, including PDF dash arrays;
   - "stroke-to-outline" fills, where each segment is a filled square-pen sweep hexagon (common in ADI/Hittite datasheets);
   - thin rectangles;
   - dot squares;
   - long outline polygons.

   Pen widths are clustered per page so estimation noise can't split one curve into two styles.
2. **Axes** (`axes.py`) find plots by pairing an evenly spaced column of numeric labels (y) with a row below it (x). Tick positions are snapped onto gridlines, then a linear or log10 mapping is fitted. The residual is reported; it is typically under 0.1 % of span.
3. **Legend** (`series.py`): each text line inside the frame plus the short horizontal stroke run beside it becomes a `(label, line style)` pair. Grid and frame styles are excluded automatically because they're entirely axis-aligned.
4. **Series assignment** (`assign.py`):
   - Legend samples define a dash "grammar": element types (dot, specific dash lengths, solid) and the gap expected between them.
   - Every stroke in the plot is classified by element type, with tolerance for dashes stretched on steep segments.
   - Strokes are linked with a global min-cost assignment that only allows transitions some legend pattern permits.
   - Linked tracks are voted into legend entries, and leftover fragments are absorbed by direction-continuity growth.
   - Color and pen width also separate series.

## Reading the confidence

Each series has a `confidence` between 0 and 1. It drops when links needed loose gap matches or when fragments overlap in x, which means a series probably picked up pieces of a neighboring curve. Anything under ~0.5 is worth a look at its overlay.

Two situations are flagged rather than guessed:

- **Identical legend styles.** Some datasheets draw two series with the same line style. On HMC576 "Output Power vs. Drive Level", −6 dBm and 0 dBm are both dotted. Those curves are emitted as `"-6dBm | 0dBm [curve k of n, top to bottom]"` with `ambiguous_with` set.
- **Overlapping fragments** get a `note` and reduced confidence.

## Optional: local LLM for metadata

Point it at any OpenAI-compatible endpoint, such as llama.cpp server, vLLM, Ollama, LM Studio, or the model behind a Hermes agent box:

```
plot-extractor datasheets/ -o out/ --llm-url http://hermes.local:8080/v1 --llm-model <model-name>
```

Each plot's `metadata` then holds structured fields parsed from the title, axis labels and legend:

- quantity and unit for each axis
- the series variable and its values
- fixed test conditions (e.g. `{"drive level": "3 dBm"}`)

If the endpoint is unreachable, extraction still completes and the error is recorded.

An agent can also drive the tool directly: run the CLI, read `index.jsonl`, and open overlays for low-confidence series.

## Python API

```python
from plot_extractor import extract_pdf, interpolate

res = extract_pdf("hmc576.pdf")
for p in res.plots:
    print(p.frame.title, [s.label for s in p.series])
s = res.plots[0].series[0]
interpolate(s, [20.0, 24.0])   # y at given x
```

## Limitations / roadmap

- **Raster figures** (scanned or older datasheets) are detected and listed in `summary.json`, but not extracted yet. `raster.py` sketches the planned OpenCV + Tesseract path.
- **Dense same-color clusters.** When several dashed series are packed within a few points of each other, linking can mix them. The confidence and overlap notes catch this.
- **Axes** need at least 3 numeric tick labels on each axis. Dual-y-axis plots and plots with labels only inside the frame aren't handled yet.
- Points are the vertices the plotting software drew. For dashed curves, that means dash endpoints. Use `interpolate()` for a uniform grid.

## Tests

```
pip install -e .[dev]
pytest
```

`tests/test_synthetic.py` generates a datasheet-style PDF with known curves (solid/dashed/dotted, colored, log axis) and checks accuracy. `tests/test_hmc576.py` is a regression test on the ADI HMC576 datasheet. It runs when `tests/data/hmc576.pdf` is present; the PDF is not committed.
