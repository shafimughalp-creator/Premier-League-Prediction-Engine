"""Screenshot every screen and state in the reference and the app, then
pixel-diff them at 1440, 1024 and 390 px.

    python -m streamlit run app.py --server.port=8501
    python tests/capture.py --base-url http://127.0.0.1:8501

Artifacts (reference, app, amplified diff, side-by-side) land in
``tests/artifacts/``. The command exits non-zero when a pair differs by
more than a sub-pixel antialiasing tolerance or when the page text or
document height does not match.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

WIDTHS = {1440: 900, 1024: 768, 390: 844}

# name, route, fixture id, action
STATES = [
    ("overview", "overview", None, None),
    ("overview_hover", "overview", None, "hover"),
    ("week", "week", None, None),
    ("week_home", "week", None, "chip:home"),
    ("week_draw", "week", None, "chip:draw"),
    ("week_away", "week", None, "chip:away"),
    ("table", "table", None, None),
    ("table_drawer", "table", None, "drawer"),
    ("match_liverpool", "match", common.MATCH_IDS[0], None),
    ("match_newcastle", "match", common.MATCH_IDS[1], None),
    ("match_tottenham", "match", common.MATCH_IDS[2], None),
]

TEXT_JS = """
() => ['#side', '#main', '#drawer'].map(sel => {
  const el = document.querySelector(sel);
  return el ? el.innerText.replace(/\\s+/g, ' ').trim() : '';
}).join(' | ')
"""


def do_action(page, action: str | None) -> None:
    if not action:
        return
    if action.startswith("chip:"):
        page.click(f'[data-f="{action.split(":", 1)[1]}"]')
        page.wait_for_timeout(1100)
        page.evaluate(common.SCROLL_ALL)
    elif action == "drawer":
        page.click("tbody tr[data-team]")
        page.wait_for_timeout(350)
    elif action == "menu":
        page.click("#menuBtn")
        page.wait_for_timeout(350)
    elif action == "hover":
        page.hover(".grid .fx")


def run(base_url: str, widths: list[int], only: list[str], out: Path) -> int:
    failures = 0
    out.mkdir(parents=True, exist_ok=True)

    with common.reference_server() as ref_base, sync_playwright() as p:
        browser = common.launch(p)
        for width in widths:
            height = WIDTHS.get(width, 900)
            context = browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=1,
                reduced_motion="no-preference",
            )
            states = [s for s in STATES if not only or s[0] in only]
            if width < 1024:
                states = states + [("overview_menu", "overview", None, "menu")]

            print(f"\n=== {width}x{height} ===")
            for name, route, fixture_id, action in states:
                prefix = out / f"{width}_{name}"
                ref_page = context.new_page()
                app_page = context.new_page()
                errors: list[str] = []
                app_page.on(
                    "console",
                    lambda msg: errors.append(msg.text) if msg.type == "error" else None,
                )

                common.goto(
                    ref_page,
                    common.ref_url(ref_base, route, fixture_id),
                    common.TITLES[route],
                )
                common.goto(
                    app_page,
                    common.app_url(base_url, route, fixture_id),
                    common.TITLES[route],
                )

                common.settle(ref_page)
                common.settle(app_page)

                do_action(ref_page, action)
                do_action(app_page, action)

                ref_text = ref_page.evaluate(TEXT_JS)
                app_text = app_page.evaluate(TEXT_JS)
                ref_height = ref_page.evaluate("document.documentElement.scrollHeight")
                app_height = app_page.evaluate("document.documentElement.scrollHeight")

                ref_path = common.shot(ref_page, prefix.with_name(prefix.name + "_ref.png"))
                app_path = common.shot(app_page, prefix.with_name(prefix.name + "_app.png"))
                result = common.diff(ref_path, app_path, prefix)

                text_ok = ref_text == app_text
                height_ok = ref_height == app_height
                pixel_ok = (
                    result["same_size"]
                    and (result["mean"] or 0) <= 0.05
                    and (result["max"] or 0) <= 32
                )
                ok = pixel_ok and text_ok and height_ok and not errors
                failures += 0 if ok else 1

                print(
                    f"{'PASS' if ok else 'FAIL'} {name:<18} "
                    f"{result['size_ref']} max={result['max']} mean={result['mean']} "
                    f"gt10={result['gt10']} text={'ok' if text_ok else 'DIFF'} "
                    f"height={'ok' if height_ok else f'{ref_height}!={app_height}'}"
                )
                if not text_ok:
                    first = next(
                        (
                            i
                            for i, (a, b) in enumerate(zip(ref_text, app_text))
                            if a != b
                        ),
                        min(len(ref_text), len(app_text)),
                    )
                    print(f"     text differs at {first}:")
                    print(f"       ref: {ref_text[max(0, first - 60):first + 100]!r}")
                    print(f"       app: {app_text[max(0, first - 60):first + 100]!r}")
                if errors:
                    print(f"     console errors: {errors}")

                ref_page.close()
                app_page.close()

            context.close()
        browser.close()

    print(f"\n{failures} failing pair(s); artifacts in {out}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8501")
    parser.add_argument("--widths", default="1440,1024,390")
    parser.add_argument("--state", default="", help="comma-separated state names")
    parser.add_argument("--out", default=str(common.ARTIFACTS))
    args = parser.parse_args()
    widths = [int(w) for w in args.widths.split(",") if w.strip()]
    only = [s for s in args.state.split(",") if s.strip()]
    return run(args.base_url, widths, only, Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
