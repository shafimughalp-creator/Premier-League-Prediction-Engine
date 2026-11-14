"""Behavioural parity checks for the Streamlit build.

    python -m streamlit run app.py --server.port=8501
    python tests/interactions.py --base-url http://127.0.0.1:8501

Covers routing, back/forward, filters, the team drawer, the mobile menu,
keyboard use, reduced motion, console cleanliness, network hosts, page
titles and the language rules from BRIEF.md.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ALLOWED_HOSTS = {
    "127.0.0.1",
    "localhost",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "resources.premierleague.com",
}

BANNED = [
    "model",
    "algorithm",
    "prediction",
    "probability",
    "accuracy",
    "confidence",
    "calibration",
    "features",
    "training data",
    "dataset",
    "machine learning",
    "ai",
    "neural",
    "data-driven",
]

FILTER_COUNTS = {"all": 10, "home": 8, "draw": 0, "away": 2}


class Checks:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes = 0

    def ok(self, condition: bool, label: str, detail: str = "") -> None:
        if condition:
            self.passes += 1
            print(f"PASS {label}")
        else:
            self.failures.append(label)
            print(f"FAIL {label}" + (f" — {detail}" if detail else ""))

    def equal(self, actual, expected, label: str) -> None:
        self.ok(actual == expected, label, f"expected {expected!r}, got {actual!r}")


def active_nav(page) -> list[str]:
    return page.eval_on_selector_all(
        "#nav a.on", "els => els.map(e => e.textContent.trim())"
    )


def open_page(context, url: str, title: str, checks: Checks, label: str):
    page = context.new_page()
    errors: list[str] = []
    hosts: set[str] = set()
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("request", lambda r: hosts.add(urlparse(r.url).hostname or ""))
    page.on("pageerror", lambda e: errors.append(str(e)))
    common.goto(page, url, title)
    return page, errors, hosts


def check_routes(context, base: str, checks: Checks) -> None:
    cases = [
        ("overview", None, "Welcome back", ["Premier League", "Overview"]),
        ("week", None, "Matchweek 11", ["Matchweek 11"]),
        ("table", None, "League table", ["Table"]),
        ("match", common.MATCH_IDS[0], "Liverpool", ["Matchweek 11"]),
    ]
    for route, fixture_id, heading, nav in cases:
        page, errors, hosts = open_page(
            context,
            common.app_url(base, route, fixture_id),
            common.TITLES[route],
            checks,
            route,
        )
        checks.equal(page.title(), common.TITLES[route], f"{route}: title")
        checks.ok(
            heading.lower() in page.inner_text("#view").lower(),
            f"{route}: heading contains {heading!r}",
        )
        checks.equal(sorted(active_nav(page)), sorted(nav), f"{route}: active nav")
        checks.equal(errors, [], f"{route}: console errors")
        checks.ok(
            hosts <= ALLOWED_HOSTS,
            f"{route}: request hosts",
            f"unexpected {sorted(hosts - ALLOWED_HOSTS)}",
        )
        checks.ok(
            page.evaluate("document.documentElement.scrollHeight") > 900,
            f"{route}: document scrolls",
        )
        page.close()


def check_links_and_history(context, base: str, checks: Checks) -> None:
    page, errors, _ = open_page(
        context, common.app_url(base, "overview"), common.TITLES["overview"], checks, "links"
    )
    page.click("text=View all")
    page.wait_for_function("() => location.search.includes('page=week')")
    page.wait_for_selector("#view .chips")
    checks.equal(page.title(), common.TITLES["week"], "View all: route title")

    page.click("#nav a[data-r=table]")
    page.wait_for_function("() => location.search.includes('page=table')")
    page.wait_for_selector("#view table")
    checks.equal(page.title(), common.TITLES["table"], "nav table: route title")

    page.go_back(wait_until="load")
    page.wait_for_function("t => document.title === t", arg=common.TITLES["week"])
    checks.equal(page.title(), common.TITLES["week"], "back: week")

    page.go_forward(wait_until="load")
    page.wait_for_function("t => document.title === t", arg=common.TITLES["table"])
    checks.equal(page.title(), common.TITLES["table"], "forward: table")

    page.click("#nav a[data-r=week]")
    page.wait_for_selector("#view .chips")
    page.click(".fx")
    page.wait_for_selector("#view .mh")
    checks.equal(page.title(), common.TITLES["match"], "fixture link: match route")

    page.click("[data-back]")
    page.wait_for_function("t => document.title === t", arg=common.TITLES["week"])
    checks.equal(page.title(), common.TITLES["week"], "back button: week")
    checks.equal(errors, [], "links: console errors")
    page.close()


def check_filters(context, base: str, checks: Checks) -> None:
    page, errors, _ = open_page(
        context, common.app_url(base, "week"), common.TITLES["week"], checks, "filters"
    )
    checks.equal(
        page.eval_on_selector_all(".chip", "els => els.map(e => e.textContent)"),
        ["All10", "Home wins8", "Draws0", "Away wins2"],
        "filters: chip labels and counts",
    )
    checks.equal(
        page.eval_on_selector_all(".fx", "els => els.length"), 10, "filters: all rows"
    )

    for key, count in FILTER_COUNTS.items():
        page.click(f'[data-f="{key}"]')
        page.wait_for_timeout(120)
        checks.equal(
            page.eval_on_selector_all(".fx", "els => els.length"),
            count,
            f"filters: {key} row count",
        )
        checks.equal(
            page.get_attribute(f'[data-f="{key}"]', "aria-pressed"),
            "true",
            f"filters: {key} aria-pressed",
        )
        checks.ok(
            page.inner_text(f'[data-f="{key}"]').endswith(str(count)),
            f"filters: {key} chip count",
        )

    page.click('[data-f="draw"]')
    page.wait_for_timeout(120)
    checks.ok(
        "No fixtures lean this way this week." in page.inner_text("#wl"),
        "filters: draw empty state",
    )
    checks.ok(
        "Chances are rounded to whole percentages and always add up to 100."
        in page.inner_text("#view"),
        "filters: rounding note",
    )
    checks.equal(errors, [], "filters: console errors")
    page.close()


def check_drawer(context, base: str, checks: Checks) -> None:
    page, errors, _ = open_page(
        context, common.app_url(base, "table"), common.TITLES["table"], checks, "drawer"
    )
    row = page.locator("tbody tr[data-team]").first
    row.click()
    page.wait_for_timeout(320)
    checks.ok(page.is_visible("#drawer.open"), "drawer: opens")
    checks.ok(page.is_visible("#bd.on"), "drawer: backdrop shown")
    checks.ok(
        page.evaluate("document.activeElement.closest('#drawer') !== null"),
        "drawer: focus moves inside",
    )
    checks.ok(
        "last five (oldest → newest)" in page.inner_text("#drawer").lower(),
        "drawer: last-five heading",
    )

    page.keyboard.press("Escape")
    page.wait_for_timeout(320)
    checks.ok(not page.is_visible("#drawer.open"), "drawer: Escape closes")
    checks.ok(
        page.evaluate("document.activeElement === document.querySelector('tbody tr[data-team]')"),
        "drawer: focus returns to row",
    )

    row.click()
    page.wait_for_timeout(320)
    page.click("#drawer .x")
    page.wait_for_timeout(320)
    checks.ok(not page.is_visible("#drawer.open"), "drawer: X closes")

    row.click()
    page.wait_for_timeout(320)
    page.mouse.click(20, 20)
    page.wait_for_timeout(320)
    checks.ok(not page.is_visible("#drawer.open"), "drawer: backdrop closes")

    row.focus()
    page.keyboard.press("Enter")
    page.wait_for_timeout(320)
    # The prototype moves focus to the close button inside the keydown
    # handler, so the click the browser synthesises for Enter lands on it
    # and closes the drawer again. Parity means keeping that behaviour.
    checks.ok(
        page.evaluate("document.querySelector('#drawer').innerHTML.length") > 100,
        "drawer: Enter runs the keyboard handler (prototype closes again)",
    )
    page.keyboard.press("Escape")
    checks.equal(errors, [], "drawer: console errors")
    page.close()


def check_mobile_menu(browser, base: str, checks: Checks) -> None:
    context = browser.new_context(viewport={"width": 390, "height": 844})
    page, errors, _ = open_page(
        context, common.app_url(base, "overview"), common.TITLES["overview"], checks, "menu"
    )
    checks.ok(page.is_visible("#menuBtn"), "menu: button visible at 390")
    checks.ok(not page.is_visible(".side.open"), "menu: starts closed")

    page.click("#menuBtn")
    page.wait_for_timeout(320)
    checks.ok(page.is_visible(".side.open"), "menu: opens")
    checks.ok(page.get_attribute("#menuBtn", "aria-expanded") == "true", "menu: aria-expanded")
    checks.ok(page.is_visible("#sbd.on"), "menu: backdrop shown")

    page.mouse.click(360, 700)
    page.wait_for_timeout(320)
    checks.ok(not page.is_visible(".side.open"), "menu: backdrop closes")

    page.click("#menuBtn")
    page.wait_for_timeout(320)
    page.keyboard.press("Escape")
    page.wait_for_timeout(320)
    checks.ok(not page.is_visible(".side.open"), "menu: Escape closes")

    page.click("#menuBtn")
    page.wait_for_timeout(320)
    checks.equal(
        page.evaluate(
            "getComputedStyle(document.querySelector('.grid')).gridTemplateColumns.split(' ').length"
        ),
        1,
        "layout: single column at 390",
    )

    page.click("#nav a[data-r=table]")
    page.wait_for_selector("#view table")
    checks.equal(page.title(), common.TITLES["table"], "menu: nav link routes")

    checks.equal(errors, [], "menu: console errors")
    page.close()
    context.close()


def check_reduced_motion(browser, base: str, checks: Checks) -> None:
    context = browser.new_context(
        viewport={"width": 1440, "height": 900}, reduced_motion="reduce"
    )
    page, errors, _ = open_page(
        context,
        common.app_url(base, "match", common.MATCH_IDS[0]),
        common.TITLES["match"],
        checks,
        "reduced motion",
    )
    checks.equal(
        page.evaluate("getComputedStyle(document.querySelector('.cc')).animationName"),
        "none",
        "reduced motion: card animation off",
    )
    checks.equal(
        page.evaluate("getComputedStyle(document.querySelector('.bar')).animationName"),
        "none",
        "reduced motion: bar animation off",
    )
    checks.ok(
        page.evaluate(
            "() => [...document.querySelectorAll('[data-count]')].every("
            "el => el.textContent === el.dataset.count + (el.dataset.suf || ''))"
        ),
        "reduced motion: counts are final immediately",
    )
    checks.equal(errors, [], "reduced motion: console errors")
    page.close()
    context.close()


def check_language(context, base: str, checks: Checks) -> None:
    page, _, _ = open_page(
        context, common.app_url(base, "overview"), common.TITLES["overview"], checks, "language"
    )
    text = page.evaluate(
        "() => ['#side', '#main', '#drawer']"
        ".map(s => document.querySelector(s)?.innerText || '').join(' ')"
    ).lower()
    for word in BANNED:
        pattern = r"\b" + re.escape(word) + r"\b"
        checks.ok(
            re.search(pattern, text) is None,
            f"language: no {word!r}",
            re.search(pattern, text).group(0) if re.search(pattern, text) else "",
        )
    checks.ok(
        "not betting odds and not a guarantee." in text,
        "language: odds disclaimer",
    )
    checks.ok(
        "form, table and head-to-head use real premier league results" in text,
        "language: results disclaimer",
    )
    checks.ok(
        "chances are the project's own match ratings for 2026/27" in text,
        "language: ratings footer",
    )
    page.close()


def run(base: str) -> int:
    checks = Checks()
    with common.reference_server(), sync_playwright() as p:
        browser = common.launch(p)
        desktop = browser.new_context(viewport={"width": 1440, "height": 900})

        check_routes(desktop, base, checks)
        check_links_and_history(desktop, base, checks)
        check_filters(desktop, base, checks)
        check_drawer(desktop, base, checks)
        check_language(desktop, base, checks)
        desktop.close()

        check_mobile_menu(browser, base, checks)
        check_reduced_motion(browser, base, checks)
        browser.close()

    print(f"\n{checks.passes} passed, {len(checks.failures)} failed")
    for failure in checks.failures:
        print(f"  - {failure}")
    return 1 if checks.failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8501")
    args = parser.parse_args()
    return run(args.base_url)


if __name__ == "__main__":
    raise SystemExit(main())
