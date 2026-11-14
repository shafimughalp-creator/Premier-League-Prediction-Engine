# EPL Intelligence — Streamlit match centre

A Streamlit build of the approved `reference/index.html` match centre for the
English Premier League, 2026/27, Matchweek 11. The prototype is the source of
truth: layout, copy, colours, badges, animations, responsive behaviour,
routing and interactions are reproduced unchanged.

## Run

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the URL Streamlit prints (usually <http://localhost:8501>).

The Streamlit CLI is not always on `PATH`; `python -m streamlit run app.py`
works from any shell.

## Routes

| Screen | URL |
| --- | --- |
| Overview | `?page=overview` (also the default at `/`) |
| Matchweek 11 | `?page=week` |
| Match detail | `?page=match&id=2026-11-21_Liverpool_Man%20United` |
| League table | `?page=table` |

Routing uses `st.query_params`, so links are real navigations, the browser
back/forward buttons work and the sidebar marks the active section on load.

## Files

```
app.py              entry point: page config, query routing, one st.html call
data.py             DATA (meta, seasonNumbers, table, fixtures, club badges, logos)
render.py           Python ports of the prototype's overview/week/match/table/shell renderers
static/style.css    the prototype CSS, copied verbatim
static/shell.css    Streamlit chrome removal, document scrolling, Inter import
static/app.js       count-ups, scroll reveal, badge fallback, filters, drawer, menu
requirements.txt    runtime and test dependencies
reference/index.html locked approved prototype (never edited)
tests/              Playwright screenshot, diff and interaction checks
PARITY.md           reference → Streamlit inventory, ticked after verification
```

## How it works

* `app.py` reads `page` and `id` from `st.query_params`, asks `render.py` for
  the matching screen, then injects the prototype CSS, the shell CSS, the
  rendered HTML and `static/app.js` in a single `st.html(...,
  unsafe_allow_javascript=True)` call. That keeps the markup in the main
  document, so the prototype's global selectors, fixed sidebar and fixed
  drawer behave exactly as they do in `reference/index.html`.
* `static/shell.css` hides Streamlit's header, toolbar, footer, menu, sidebar
  and padding, makes the document (not Streamlit's main pane) the scroll
  container, and hands font/colour/line-height back to the prototype rules.
* `static/app.js` supplies only the behaviour Streamlit cannot express:
  count-ups, IntersectionObserver reveal, the badge fallback (Streamlit strips
  inline `onerror`), client-side filter chips, the team drawer, the mobile
  menu, Escape/Enter handling and per-route document titles. It also
  re-hydrates two things Streamlit's sanitizer strips: the SVG `<defs>`/brand
  mark and the `<!-- TODO: real value -->` comments that follow every "—"
  placeholder (rendered as `<i data-miss>` and swapped for real comment nodes
  so text shaping matches the prototype).
* All displayed numbers come from `data.py`. Dates, fixture counts, kick-off
  times and season labels are derived from `DATA`, not typed into markup.

## Tests

The Playwright suite needs the Chromium browser once:

```bash
python -m playwright install chromium
```

Start the app on a known port, then run the checks:

```bash
python -m streamlit run app.py --server.headless=true --server.port=8801
python tests/capture.py --base-url http://127.0.0.1:8801
python tests/interactions.py --base-url http://127.0.0.1:8801
```

`capture.py` screenshots every screen/state from `reference/index.html` and
the running app at 1440, 1024 and 390 px, writes pixel diffs to
`tests/artifacts/`, and prints a per-pair difference summary. `interactions.py`
checks routing, back/forward, filters, drawer open/close/Escape, the mobile
menu, keyboard use, reduced motion, console errors and network hosts.

## Notes

* The approved design is reproduced as-is. Possible improvements are listed
  in the final hand-off rather than applied.
* `reference/index.html` is locked and is never modified.
