"""Tests for resume upload helpers in app.py (local data only, no API calls)."""
from pathlib import Path
from unittest.mock import patch

import pytest

APP = Path(__file__).resolve().parent.parent / "app.py"


def _load_helpers():
    source = APP.read_text(encoding="utf-8")
    start = source.index("MAX_UPLOAD_BYTES")
    end = source.index("FORM_KEYS = (")
    namespace = {}
    exec("import io\nfrom typing import Optional, Tuple\n" + source[start:end], namespace)
    return namespace


H = _load_helpers()
extract_resume_text = H["extract_resume_text"]
resolve_resume_input = H["resolve_resume_input"]
ResumeFileError = H["ResumeFileError"]


def make_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


def test_txt_extraction_utf8_and_cp1252():
    assert extract_resume_text("cv.txt", "  Skills: Python, SQL \n".encode()) == "Skills: Python, SQL"
    assert extract_resume_text("CV.TXT", "Caf\xe9 manager".encode("cp1252")) == "Café manager"


def test_pdf_extraction():
    assert "Python and SQL" in extract_resume_text("cv.pdf", make_pdf("Python and SQL"))


def test_empty_txt_and_blank_pdf_rejected():
    with pytest.raises(ResumeFileError, match="No text"):
        extract_resume_text("cv.txt", b"   \n ")
    with pytest.raises(ResumeFileError, match="No text"):
        extract_resume_text("cv.pdf", make_pdf(""))


def test_invalid_pdf_unsupported_type_and_oversize_are_friendly():
    with pytest.raises(ResumeFileError, match="could not be read"):
        extract_resume_text("cv.pdf", b"this is not a pdf")
    with pytest.raises(ResumeFileError, match="Unsupported"):
        extract_resume_text("cv.docx", b"x")
    with pytest.raises(ResumeFileError, match="too large"):
        extract_resume_text("cv.txt", b"a" * (5 * 1024 * 1024 + 1))


def test_both_upload_and_pasted_text_is_rejected():
    text, error = resolve_resume_input("cv.txt", b"Python", "pasted resume")
    assert text is None and "only one" in error


def test_upload_only_pasted_only_and_neither():
    assert resolve_resume_input("cv.txt", b"Python", "  ") == ("Python", None)
    assert resolve_resume_input(None, None, " pasted ") == ("pasted", None)
    assert resolve_resume_input(None, None, "") == (None, None)
    text, error = resolve_resume_input("cv.pdf", b"junk", "")
    assert text is None and error


def test_pasted_text_reaches_pipeline_unchanged():
    from streamlit.testing.v1 import AppTest
    import orchestration

    with patch.object(orchestration, "run_pipeline", return_value=None) as mocked:
        at = AppTest.from_file(str(APP), default_timeout=20).run()
        at.text_area[0].set_value("My resume text")
        at.text_area[1].set_value("Job description")
        at.button[0].click().run()
    assert not at.exception
    assert mocked.call_args.args[0] == "My resume text"
