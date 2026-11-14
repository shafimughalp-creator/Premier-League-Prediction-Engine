"""Shared helpers for the Playwright parity checks.

The reference is a static file with hash routes (``#/week``); the
Streamlit app uses query routes (``?page=week``). Everything else — the
markup, the CSS and the interactions — should be identical, so each
check pairs a reference page with an app page at the same viewport.
"""

from __future__ import annotations

import contextlib
import functools
import http.server
import threading
from pathlib import Path
from urllib.parse import quote, urlencode

from PIL import Image, ImageChops
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "tests" / "artifacts"

TITLES = {
    "overview": "Overview · EPL Intelligence",
    "week": "Matchweek 11 · EPL Intelligence",
    "match": "Match · EPL Intelligence",
    "table": "Table · EPL Intelligence",
}

# Fixtures used for match-detail comparisons: the featured home pick, an
# away pick and a different home pick.
MATCH_IDS = [
    "2026-11-21_Liverpool_Man United",
    "2026-11-21_Newcastle_Arsenal",
    "2026-11-21_Tottenham_Ipswich",
]

FREEZE_CSS = (
    "*,*::before,*::after{animation:none!important;"
    "transition:none!important;scroll-behavior:auto!important}"
)

SCROLL_ALL = """
async () => {
  document.documentElement.style.scrollBehavior = 'auto';
  const h = document.documentElement.scrollHeight;
  for (let y = 0; y <= h; y += 300) {
    window.scrollTo(0, y);
    await new Promise(r => requestAnimationFrame(() => setTimeout(r, 20)));
  }
  window.scrollTo(0, h);
  await new Promise(r => setTimeout(r, 250));
  window.scrollTo(0, 0);
  await new Promise(r => setTimeout(r, 100));
}
"""


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D401 - silence the reference server
        pass


@contextlib.contextmanager
def reference_server():
    """Serve the repository so ``reference/index.html`` loads over HTTP."""
    handler = functools.partial(_QuietHandler, directory=str(ROOT))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def launch(p):
    """Launch Chromium, falling back to an installed Chrome/Edge."""
    attempts = [
        ("bundled chromium", {}),
        ("Microsoft Edge", {"channel": "msedge"}),
        ("Google Chrome", {"channel": "chrome"}),
    ]
    problems = []
    for label, kwargs in attempts:
        try:
            return p.chromium.launch(**kwargs)
        except Exception as exc:  # pragma: no cover - depends on the machine
            problems.append(f"{label}: {exc}")
    raise RuntimeError(
        "No Chromium available. Run `python -m playwright install chromium` "
        "or install Google Chrome / Microsoft Edge.\n" + "\n".join(problems)
    )


def app_url(base: str, page: str, fixture_id: str | None = None) -> str:
    query = {"page": page}
    if fixture_id:
        query["id"] = fixture_id
    return base.rstrip("/") + "/?" + urlencode(query)


def ref_url(base: str, page: str, fixture_id: str | None = None) -> str:
    if page == "match":
        route = "/match/" + quote(fixture_id or "", safe="")
    elif page == "week":
        route = "/week"
    elif page == "table":
        route = "/table"
    else:
        route = "/"
    return base.rstrip("/") + "/reference/index.html#" + route


def goto(page, url: str, title: str) -> None:
    page.goto(url, wait_until="load")
    page.wait_for_function("t => document.title === t", arg=title, timeout=30000)
    page.wait_for_selector("#view .head, #view .mh", state="attached", timeout=30000)


def settle(page) -> None:
    """Freeze motion, finish count-ups, reveal lazy sections, load images."""
    page.add_style_tag(content=FREEZE_CSS)
    page.wait_for_timeout(1200)
    with contextlib.suppress(Exception):
        page.wait_for_function(
            "() => [...document.querySelectorAll('[data-count]')].every("
            "el => el.textContent === el.dataset.count + (el.dataset.suf || ''))",
            timeout=3000,
        )
    page.evaluate(SCROLL_ALL)
    page.evaluate("() => Promise.race([document.fonts.ready, new Promise(r => setTimeout(r, 3000))])")
    page.evaluate(
        "() => Promise.race(["
        "Promise.all([...document.images].map(i => i.decode().catch(() => {}))),"
        "new Promise(r => setTimeout(r, 4000))])"
    )
    page.wait_for_timeout(100)


def shot(page, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(
        path=str(path),
        full_page=True,
        animations="disabled",
        caret="hide",
    )
    return path


def diff(ref_path: Path, app_path: Path, out_prefix: Path) -> dict:
    """Pixel-diff two screenshots and write an amplified diff + side-by-side."""
    ref = Image.open(ref_path).convert("RGB")
    app = Image.open(app_path).convert("RGB")
    result = {
        "size_ref": ref.size,
        "size_app": app.size,
        "same_size": ref.size == app.size,
        "max": None,
        "mean": None,
        "gt10": None,
        "gt30": None,
        "bbox": None,
        "diff_path": None,
    }
    if ref.size != app.size:
        return result

    delta = ImageChops.difference(ref, app)
    arr = np.asarray(delta).astype(np.uint16)
    result["bbox"] = delta.getbbox()
    result["max"] = int(arr.max())
    result["mean"] = round(float(arr.mean()), 4)
    result["gt10"] = int((arr.max(axis=2) > 10).sum())
    result["gt30"] = int((arr.max(axis=2) > 30).sum())

    if result["bbox"]:
        amplified = Image.fromarray(np.clip(arr * 8, 0, 255).astype(np.uint8))
        side = Image.new("RGB", (ref.width * 2 + 12, ref.height), (20, 20, 20))
        side.paste(ref, (0, 0))
        side.paste(app, (ref.width + 12, 0))
        side.save(out_prefix.with_name(out_prefix.name + "_side.png"))
        amplified.save(out_prefix.with_name(out_prefix.name + "_amp.png"))
        result["diff_path"] = str(out_prefix.with_name(out_prefix.name + "_amp.png"))
    return result
