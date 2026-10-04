"""Converts doc/README.md into a styled doc/README.html, which is then
printed to doc/README.pdf using headless Chrome/Edge (--print-to-pdf).

Not wired into any pipeline - run manually whenever the documentation changes:
    python scripts/build_doc_pdf.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

import markdown

BASE_DIR = Path(__file__).resolve().parent.parent
DOC_DIR = BASE_DIR / "doc"
MD_PATH = DOC_DIR / "README.md"
HTML_PATH = DOC_DIR / "README.html"
PDF_PATH = DOC_DIR / "README.pdf"

CSS = """
<style>
  @page { margin: 2cm 2cm 2.2cm 2cm; }
  body {
    font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
    color: #1a1a1a;
    line-height: 1.55;
    font-size: 11.5pt;
    max-width: 880px;
    margin: 0 auto;
  }
  h1 { font-size: 22pt; border-bottom: 3px solid #2b6cb0; padding-bottom: 8px; margin-top: 0; }
  h2 { font-size: 16pt; color: #1a4971; border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top: 32px; page-break-after: avoid; }
  h3 { font-size: 13pt; color: #1a4971; margin-top: 22px; page-break-after: avoid; }
  code {
    background: #f1f3f5;
    padding: 1px 5px;
    border-radius: 4px;
    font-family: Consolas, "Courier New", monospace;
    font-size: 10pt;
  }
  pre {
    background: #282c34;
    color: #e6e6e6;
    padding: 12px 14px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 9.5pt;
    page-break-inside: avoid;
  }
  pre code { background: none; color: inherit; padding: 0; }
  table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 10.5pt; }
  th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
  th { background: #e9eef5; }
  img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 8px 0; page-break-inside: avoid; }
  blockquote {
    border-left: 4px solid #2b6cb0;
    margin: 12px 0;
    padding: 4px 14px;
    background: #f5f8fb;
    color: #333;
  }
  a { color: #2b6cb0; text-decoration: none; }
  hr { border: none; border-top: 1px solid #ccc; margin: 24px 0; }
  ul, ol { margin: 6px 0; }
</style>
"""


def find_browser() -> str:
    candidates = [
        shutil.which("msedge"),
        shutil.which("chrome"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path
    raise FileNotFoundError("No Edge/Chrome executable found to render the PDF.")


def main():
    md_text = MD_PATH.read_text(encoding="utf-8")
    body_html = markdown.markdown(
        md_text,
        extensions=["extra", "tables", "sane_lists", "toc"],
    )
    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{CSS}</head><body>{body_html}</body></html>"
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {HTML_PATH}")

    browser = find_browser()
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        f"--print-to-pdf={PDF_PATH}",
        "--no-pdf-header-footer",
        "--print-to-pdf-no-header",
        HTML_PATH.resolve().as_uri(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0 or not PDF_PATH.exists():
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {PDF_PATH} ({PDF_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
