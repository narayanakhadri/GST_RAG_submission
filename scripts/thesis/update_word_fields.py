"""Update every field of a thesis .docx in Microsoft Word itself: TOC, list of tables, list of figures, figure/table
numbers and cross-references, so the file opens correct on any machine without the "update fields?" prompt.

    uv run python -m scripts.thesis.update_word_fields docs/thesis/GST_RAG_Thesis_v1.docx [more.docx ...]

python-docx can write fields but not compute them (page numbers need Word's layout), and build_thesis.py only asks
Word to update them on open (w:updateFields), which Protected View or a "No" click skips. This drives Word through
COM (Windows, Word installed; the file must not be open in Word), updates the fields twice so page numbers settle
after the lists grow, saves, and then removes w:updateFields so Word no longer prompts.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

PS = r"""
$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
  foreach ($p in $args) {
    $doc = $word.Documents.Open($p, $false, $false, $false)
    foreach ($pass in 1..2) {
      $doc.Fields.Update() | Out-Null
      foreach ($t in $doc.TablesOfContents) { $t.Update() | Out-Null }
      foreach ($t in $doc.TablesOfFigures) { $t.Update() | Out-Null }
    }
    "{0}: {1} fields, {2} TOC, {3} lists, {4} pages" -f $p, $doc.Fields.Count, $doc.TablesOfContents.Count, $doc.TablesOfFigures.Count, $doc.ComputeStatistics(2)
    $doc.Save()
    $doc.Close()
  }
} finally {
  $word.Quit()
  [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
"""


def drop_update_fields(path: Path) -> None:
    """Remove <w:updateFields/> from word/settings.xml (the fields are already current)."""
    tmp = path.with_suffix(".tmp.docx")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/settings.xml":
                data = re.sub(rb"<w:updateFields[^>]*/>", b"", data)
            zout.writestr(item, data)
    tmp.replace(path)


def update(paths: list[Path]) -> None:
    for p in paths:
        lock = p.with_name("~$" + p.name[2:])
        if lock.exists():
            raise SystemExit(f"{p} is open in Word ({lock.name} exists): close it first")
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8-sig") as f:
        f.write(PS)
        script = f.name
    out = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script,
                          *[str(p.resolve()) for p in paths]], capture_output=True, text=True)
    Path(script).unlink(missing_ok=True)
    print(out.stdout.strip())
    if out.returncode != 0:
        raise SystemExit(out.stderr.strip())
    for p in paths:
        drop_update_fields(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("docx", type=Path, nargs="+")
    update(ap.parse_args().docx)


if __name__ == "__main__":
    main()
