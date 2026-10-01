"""Record the README demo (animated WebP) and screenshots from the running app, using a headless browser.

  .venv/bin/python -m playwright install chromium      # once
  PHOTOFIX_PUBLIC=1 .venv/bin/uvicorn server.main:app --port 8000 &
  .venv/bin/python -m scripts.record_demo

Writes docs/media/demo.webp and docs/media/{home,result,result-dark,pro,mobile}.png. Uses only the
bundled CC0 example photos, so everything recorded is safe to publish.
"""

import argparse
import io
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

OUT = Path("docs/media")
VIEWPORT = {"width": 1240, "height": 780}
GIF_WIDTH = 900


class Recorder:
    def __init__(self, page: Page):
        self.page, self.frames = page, []  # (image, duration ms)

    def snap(self, hold_ms: int = 120):
        img = Image.open(io.BytesIO(self.page.screenshot())).convert("RGB")
        img = img.resize((GIF_WIDTH, round(img.height * GIF_WIDTH / img.width)), Image.Resampling.LANCZOS)
        self.frames.append((img, hold_ms))

    def save(self, path: Path):
        """Animated WebP: full color (no GIF palette banding on photos) at a fraction of a GIF's size."""
        images, durations = zip(*self.frames)
        images[0].save(path, save_all=True, append_images=list(images[1:]), duration=list(durations),
                       loop=0, quality=82, method=6)


def open_example(page: Page, label: str):
    if page.is_visible("#reset"):
        page.click("#reset")
    page.click(f".example:has-text('{label}')")
    page.wait_for_selector("#result:not([hidden])", timeout=120_000)
    page.wait_for_timeout(300)


def set_split(page: Page, pct: float):
    page.eval_on_selector("#split", "(el, v) => { el.value = v; el.dispatchEvent(new Event('input')); }", pct)


def set_style(page: Page, style: str):
    page.click(f"#styles button[data-style={style}]")
    page.wait_for_function("!document.getElementById('compare').classList.contains('loading')", timeout=120_000)
    page.wait_for_timeout(250)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000/")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, color_scheme="light", device_scale_factor=1)
        page.goto(args.url)
        page.wait_for_function("[...document.querySelectorAll('.example img')].length > 0 && "
                               "[...document.querySelectorAll('.example img')].every(i => i.complete && i.naturalWidth)")
        page.screenshot(path=OUT / "home.png")
        rec = Recorder(page)
        rec.snap(1600)

        # 1. Underexposed lake: sweep the before/after divider, then compare styles.
        open_example(page, "Mountain lake")
        rec.snap(900)
        for pct in [*range(50, 4, -5), *range(5, 96, 6), *range(95, 49, -5)]:
            set_split(page, pct)
            rec.snap(55)
        rec.snap(900)
        page.screenshot(path=OUT / "result.png")
        set_style(page, "pro")
        rec.snap(1400)
        page.screenshot(path=OUT / "pro.png")
        set_style(page, "natural")
        rec.snap(700)

        # 2. Noisy night photo: neural restorer.
        open_example(page, "City at night")
        set_split(page, 50)
        rec.snap(1800)

        # 3. A good photo should be left alone.
        open_example(page, "Coast at sunset")
        rec.snap(2000)
        rec.save(OUT / "demo.webp")

        dark = browser.new_page(viewport=VIEWPORT, color_scheme="dark")
        dark.goto(args.url)
        dark.wait_for_selector(".example")
        open_example(dark, "Diner interior")
        dark.screenshot(path=OUT / "result-dark.png")

        mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
        mobile.goto(args.url)
        mobile.wait_for_selector(".example")
        open_example(mobile, "City at night")
        mobile.screenshot(path=OUT / "mobile.png", full_page=True)
        browser.close()

    for f in sorted(OUT.iterdir()):
        print(f"{f}  {f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
