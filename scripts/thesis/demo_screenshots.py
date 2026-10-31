"""Screenshots of the demo app (scripts/thesis/demo_app.py) for the thesis: Figure 4.13 and Appendix E.

    uv run streamlit run scripts/thesis/demo_app.py --server.headless true --server.port 8599   # needs Qdrant up
    uv run --no-project --with playwright --with pillow python scripts/thesis/demo_screenshots.py

Uses the installed Chrome (playwright channel "chrome", no browser download). Question 1 is run live (Gemini, cached),
so its pipeline steps can be shown; the other two show the reported run with the closed-book panel open. Full-page
shots go to eval/results/thesis/figures/demo/raw/, the cropped figures beside them; make_thesis_figures registers the
cropped ones in figures.json.
"""

import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

OUT = Path("eval/results/thesis/figures/demo")
RAW = OUT / "raw"
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8599"
WIDTH, SCALE = 1500, 1.5
CASES = [  # (name, question set, text typed into the question box, run live?)
    ("q1_printing", "dev_hard", "Satyamev Printers", True),
    ("heeralal", "dev_hard", "M/s Heeralal", False),
    ("vedanta", "dev_hard", "Vedanta Hospital", False),
]
# figure → (shot, top, bottom, left), in pixels of the shot scaled to 714 wide; left 0 keeps the sidebar
CROPS = {
    "demo_overview": ("q1_printing_live", 0, 735, 0),
    "demo_worked_example": ("q1_printing_live", 740, 1350, 170),
    "demo_failure": ("heeralal_stored_closed", 285, 985, 170),
    "demo_rag_helped": ("vedanta_stored_closed", 195, 945, 170),
}


def pick(page, radio: str, text: str) -> None:
    page.get_by_text(radio, exact=True).first.click()
    page.wait_for_timeout(1500)
    page.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"]').first.click()
    page.keyboard.type(text)
    page.wait_for_timeout(800)
    page.keyboard.press("Enter")
    page.wait_for_timeout(3000)


def expand(page, label: str) -> None:
    page.locator('[data-testid="stExpander"] summary', has_text=label).first.click()
    page.wait_for_timeout(1200)


def shoot() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": WIDTH, "height": 4200}, device_scale_factor=SCALE)
        page.goto(URL)
        page.get_by_text("Scores for this question").wait_for(timeout=240_000)
        page.wait_for_timeout(2000)
        for name, radio, text, live in CASES:
            pick(page, radio, text)
            page.screenshot(path=RAW / f"{name}_stored.png")
            expand(page, "Closed book")
            page.screenshot(path=RAW / f"{name}_stored_closed.png")
            if live:
                page.get_by_role("button", name="Run live").click()
                page.get_by_text("Done in").first.wait_for(timeout=600_000)
                page.wait_for_timeout(2500)
                expand(page, "Pipeline steps")
                page.screenshot(path=RAW / f"{name}_live.png")
            print("shot", name, flush=True)
        browser.close()


def crop() -> None:
    k = WIDTH * SCALE / 714  # displayed px → image px
    for fig, (shot, top, bottom, left) in CROPS.items():
        im = Image.open(RAW / f"{shot}.png")
        right = 684 if left else 714
        im.crop((int(left * k), int(top * k), int(right * k), int(bottom * k))).save(OUT / f"{fig}.png")
        print("cropped", fig)


if __name__ == "__main__":
    shoot()
    crop()
