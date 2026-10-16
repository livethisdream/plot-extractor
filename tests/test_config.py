from plot_extractor import config
from plot_extractor.config import flag, parse_env_file, setting


def test_parse_env_file():
    text = """
    # comment
    PLOTX_REPLOT=false
    export PLOTX_OVERLAY = "true"
    PLOTX_LLM_URL=http://box:8080/v1   # trailing comment
    PLOTX_LLM_MODEL='my model'
    EMPTY=
    junk line
    """
    d = parse_env_file(text)
    assert d == {"PLOTX_REPLOT": "false", "PLOTX_OVERLAY": "true",
                 "PLOTX_LLM_URL": "http://box:8080/v1", "PLOTX_LLM_MODEL": "my model", "EMPTY": ""}


def test_precedence(tmp_path, monkeypatch):
    f = tmp_path / "plot-extractor.env"
    f.write_text("PLOTX_REPLOT=false\nPLOTX_LLM_URL=\n")
    monkeypatch.setenv("PLOTX_CONFIG", str(f))
    monkeypatch.delenv("PLOTX_REPLOT", raising=False)
    monkeypatch.delenv("PLOTX_OVERLAY", raising=False)
    monkeypatch.delenv("PLOTX_LLM_URL", raising=False)
    config._file_settings.cache_clear()

    assert flag("PLOTX_REPLOT") is False          # from the file
    assert flag("PLOTX_OVERLAY") is True          # not set anywhere -> default
    assert setting("PLOTX_LLM_URL") is None       # empty in file = off

    monkeypatch.setenv("PLOTX_REPLOT", "true")    # environment beats the file
    assert flag("PLOTX_REPLOT") is True
    monkeypatch.setenv("PLOTX_REPLOT", "")        # empty env falls back to the file
    assert flag("PLOTX_REPLOT") is False
