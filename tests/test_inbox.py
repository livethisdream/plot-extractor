import json
import shutil

from plot_extractor.inbox import process_inbox, summary_markdown


def test_inbox_moves_and_outputs(tmp_path, synthetic_pdf):
    inbox, done, failed, out = (tmp_path / d for d in ("unprocessed", "processed", "failed", "output"))
    (inbox / "vendorA").mkdir(parents=True)
    shutil.copy(synthetic_pdf, inbox / "vendorA" / "part1.pdf")
    (inbox / "broken.pdf").write_bytes(b"not a pdf")
    (inbox / ".gitkeep").write_text("")

    res = process_inbox(inbox, done, failed, out, overlays=False)

    assert {r["source"]: r["ok"] for r in res} == {"vendorA/part1.pdf": True, "broken.pdf": False}
    assert (done / "vendorA" / "part1.pdf").exists()
    assert (failed / "broken.pdf").exists()
    assert not list(inbox.rglob("*.pdf"))
    assert (inbox / ".gitkeep").exists()
    assert list((out / "part1").glob("*.json"))
    assert len((out / "index.jsonl").read_text().splitlines()) == 2
    assert "1/2 succeeded" in summary_markdown(res)

    # same name dropped again: the earlier processed copy is kept
    shutil.copy(synthetic_pdf, inbox / "vendorA" / "part1.pdf")
    res2 = process_inbox(inbox, done, failed, out, overlays=False)
    assert res2[0]["ok"]
    assert len(list((done / "vendorA").glob("part1*.pdf"))) == 2
    rec = json.loads((out / "index.jsonl").read_text().splitlines()[-1])
    assert rec["moved_to"].endswith(".pdf")
