"""Optional metadata normalization with a local LLM (OpenAI-compatible API).

The numbers never come from the model -- they come from the vector geometry.
The model only turns free text (title, axis labels, legend labels) into
structured fields, e.g. "Output Power vs. Temperature @ 3 dBm Drive Level" ->
{"y_quantity": "output power", "y_unit": "dBm", "series_variable": "temperature",
 "conditions": {"drive level": "3 dBm"}}.

Works with any local server that speaks /v1/chat/completions (llama.cpp server,
vLLM, Ollama, LM Studio, or whatever is serving the Hermes box). Uses only the
standard library, and fails soft: if the endpoint is unreachable the plot is
written without metadata.
"""
from __future__ import annotations

import json
import re
import urllib.request

PROMPT = """You normalize metadata for a plot extracted from an electronic component datasheet.
Return ONLY a JSON object with these keys:
  "parameter": short name of what is plotted (e.g. "output power vs temperature"),
  "x_quantity", "x_unit", "y_quantity", "y_unit",
  "series_variable": what differs between the series (e.g. "temperature") or null,
  "series_values": list of {"label": <legend label exactly as given>, "value": <number or null>, "unit": <unit or null>},
  "conditions": object of fixed test conditions from the title (e.g. {"drive level": "3 dBm"}).
Do not invent information that is not in the text.

Part: <<part>>
Title: <<title>>
X axis label: <<xlabel>>
Y axis label: <<ylabel>>
Legend labels: <<labels>>
"""


def normalize(plot_dict: dict, url: str, model: str, part: str = "", timeout: float = 60.0) -> dict:
    labels = [s["label"] for s in plot_dict["series"]]
    msg = PROMPT
    for k, v in {"part": part or "unknown", "title": plot_dict["title"],
                 "xlabel": plot_dict["x_axis"]["label"], "ylabel": plot_dict["y_axis"]["label"],
                 "labels": json.dumps(labels)}.items():
        msg = msg.replace(f"<<{k}>>", str(v))
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": msg}],
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            content = json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception as e:  # unreachable, timeout, bad response
        return {"error": f"llm request failed: {e}"}
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        return {"error": "llm returned no JSON", "raw": content[:500]}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"error": "llm returned invalid JSON", "raw": content[:500]}
