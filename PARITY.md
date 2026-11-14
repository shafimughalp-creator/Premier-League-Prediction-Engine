# PARITY.md — reference → Streamlit checklist

Source of truth: `reference/index.html` (locked, never edited).
This file is the inventory of everything that must match. Items are ticked only after
verification in the running Streamlit app.

Legend: `[ ]` not yet verified · `[x]` verified in the running app.

Status: **all items verified** — `tests/capture.py` (pixel diffs, all widths),
`tests/interactions.py` (89 behavioural checks), plus the acceptance run
recorded in the final hand-off.

---

## 1. Document shell

- [x] `<html lang="en">`, charset utf-8, viewport `width=device-width,initial-scale=1`
- [x] Tab title per route: `Overview · EPL Intelligence`, `Matchweek 11 · EPL Intelligence`,
      `Match · EPL Intelligence`, `Table · EPL Intelligence`
- [x] Inter 400/500/600/700/800 loaded from Google Fonts
- [x] `body` background: three-layer radial gradient over `#0c0013`
- [x] `body::before` fixed diagonal shapes (magenta 10%, green 6%), `sway` 16s alternate
- [x] `scroll-behavior:smooth`; `font-variant-numeric:tabular-nums`
- [x] Skip link `.skip` → `#main`, hidden until focus
- [x] Global `:focus-visible` = 2px `#00ff85`, offset 2px, radius 8px

## 2. Sidebar

- [x] Fixed, 240px, `#14001c`, right border `rgba(255,255,255,.08)`, padding 26/16
- [x] Brand: 34px green circle mark with purple football glyph + "EPL Intelligence"
      (uppercase, 900, 16px, -0.02em), margin-bottom 34
- [x] Section label "Football" (11px, 700, .14em, uppercase, `#a995b8`)
- [x] Nav: "Premier League" (top, white), "Overview", "Matchweek 11", "Table" (sub, 30px indent)
- [x] Hover: white text, `rgba(255,255,255,.05)`, `translateX(3px)`
- [x] Active `.on`: white, green gradient background, 3px green left bar
- [x] Active mapping: Overview route highlights both "Premier League" and "Overview";
      Matchweek and Match highlight "Matchweek 11"; Table highlights "Table"
- [x] Bottom card `.wk`: "This weekend" + "10 fixtures · Saturday 21 Nov · 15:00"
- [x] Freshness line: "Results up to 8 November 2026"
- [x] Nav/brand/card entrance `slidein` with the original delays
- [x] Under 1024px: sidebar off-canvas (`translateX(-100%)`), `.open` shows it
- [x] Backdrop `.sbd` fades in, click closes

## 3. Main column

- [x] `margin-left:240px`, padding 36px 44px 40px, `max-width:1500px`
- [x] Menu button `.menu` hidden ≥1024px; below shows "☰ Menu"
- [x] Footer: two exact sentences, top border, 12px `#a995b8`, line-height 1.7

## 4. Overview (`?page=overview`)

- [x] Eyebrow: "Matchweek 11 · 2026/27"
- [x] H1: "Welcome back"
- [x] Sub: "Ten Premier League fixtures this weekend, with the chances for each result and the form behind them."
- [x] Hero: purple gradient, magenta polygon right half, green+cyan slash, 22px radius
- [x] Hero content: two 84px crests, "v", "Featured fixture",
      "Liverpool v Manchester United", "Saturday, 21 November 2026 · 15:00 kick-off · Premier League"
- [x] Hero big bar (14px) + 16px percentage row + green "See the full read" button
- [x] Hero gradient `drift` 14s alternate; slash `slash` 1s; crests `float` 5s with -2.5s offset
- [x] Button sheen `sheen` 3.6s (1.4s delay), hover white + lift
- [x] Section "Matchweek 11 fixtures" + "View all"
- [x] 10 fixture rows, each: time, home name+crest (right aligned), 8px bar, percentages,
      away crest+name, pick tag
- [x] Pick tags: Home win (green), Draw (grey), Away win (cyan)
- [x] Row hover: green border + `translateX(3px)`; bar thickens to 11px; crests scale 1.14 / rotate -4deg
- [x] Row entrance `rise` staggered 0.05 → 0.455 in 0.045 steps
- [x] Side card "The table" + "Full table": top 5, "· · ·" separator, bottom 3; columns #, Team, P, Pts
- [x] Side card "Season so far": 2×2 stats (2.73 goals/game, 42 home wins, 27 draws, 31 away wins),
      green 36px numbers, count-up
- [x] Both side cards + all rows/cards reveal on scroll (`rv` → `in`)
- [x] Table rows are `role=button`, `tabindex=0`, open the team drawer

## 5. Matchweek 11 (`?page=week`)

- [x] Eyebrow "Matchweek 11 · 2026/27", H1 "Matchweek 11",
      sub "Saturday, 21 November 2026 · all kick-offs 15:00"
- [x] Four chips: All 10, Home wins 8, Draws 0, Away wins 2
- [x] Active chip: green background, `#1a0022` text; inactive `#1c0526` + border
- [x] Chip hover lift -2px; pop-in animation for counts
- [x] Clicking a chip filters the list instantly, updates `.on` + `aria-pressed`
- [x] Draws filter shows "No fixtures lean this way this week."
- [x] Note: "Chances are rounded to whole percentages and always add up to 100."
- [x] Filtered list replays the row entrance animation

## 6. Match detail (`?page=match&id=…`)

- [x] Back button "← Back to fixtures" (dark, hover green border, lift) → Matchweek 11
- [x] Header: home crest 64 + name + "4th · 18 pts"; centre "Matchweek 11" / date / "15:00 kick-off";
      away name + crest 64 + "18th · 8 pts"; crests float 6s, away offset -3s
- [x] Three chance cards; pick card green border + green percentage + "Our pick" badge (pulse 2.2s + glow 3s)
- [x] Cards: label, 64px percentage, "Win in 90 minutes" / "Points shared"
- [x] "The chances" card with 14px bar
- [x] "The read": 5 bullets, green markers, exact copy pattern, "—" where data is missing
- [x] "Form": both teams, last-five chips, PPG, scored/conceded, home/away record
- [x] "Head to head": 5 meetings for Liverpool v Man United; "Last five meetings: —" elsewhere
- [x] "Season so far": Position, Points, Goals for, Goals against, Goal difference
- [x] Liverpool v Man United values: 1.8/0.6, 1.0/0.6, 1.4/1.2, 1.2/2.2, 1.0, 40%, 1.3–1.1
- [x] Other nine fixtures show "—" for missing detail fields
- [x] Unknown id: "Fixture not found. Back to fixtures"
- [x] Card entrance `rise` with the original delays

## 7. League table (`?page=table`)

- [x] Eyebrow "2026/27 · After 10 matches", H1 "League table",
      sub "Select a club for its season summary."
- [x] 20 rows, columns #, Team, P, W, D, L, GF, GA, GD, Pts, Last 5
- [x] Top four: cyan left bar + cyan tint; bottom three: magenta left bar + magenta tint
- [x] Key: "Top four" (cyan), "Bottom three" (magenta)
- [x] Row hover tint; crest scale/tilt; rows keyboard-activatable
- [x] Row entrance stagger 0.03s

## 8. Team drawer

- [x] Slides in from right, 410px max, `#14001c`, 26px padding, 0.25s transform
- [x] Backdrop `.bd` 75% dark, click closes
- [x] X button top right, 36px circle
- [x] Header: 56px crest, team name 24px, "4th · 18 pts"
- [x] 8 stat tiles (P W D L GF GA GD Pts) with pop-in and stagger 0.16→0.44
- [x] "Last five (oldest → newest)" chips
- [x] "Home and away": Home points / Away points both "—"
- [x] Opens from table row, overview mini-table row, "In form" style clicks
- [x] Escape closes; focus moves to X then returns to the trigger
- [x] Drawer content rise animation 0.1s delay

## 9. Animations (complete list)

- [x] `rise` (0.6s cubic-bezier(.2,.8,.2,1), backwards) — head, hero, back, mh, cards, cc
- [x] `rise` 0.5s — fixture rows; 0.45s — table rows
- [x] `grow` 1s 0.25s — chance bars clip-path sweep
- [x] `float` 5s/6s — hero + match-header crests
- [x] `slash` 1s 0.2s — hero green/cyan slash
- [x] `drift` 14s alternate — hero background position
- [x] `sway` 16s alternate — body diagonal shapes
- [x] `sheen` 3.6s 1.4s — button
- [x] `pulse` 2.2s — "Our pick" badge
- [x] `glow` 3s 1s — pick card
- [x] `pop` 0.45s cubic-bezier(.3,1.6,.5,1) — form chips, drawer stats
- [x] `slidein` 0.55s — sidebar brand/section/nav/card
- [x] Staggered delays: hero .05, cards .12/.2, cc .08/.16, nav .12/.19/.26/.33,
      chips .37/.43/.49/.55, drawer stats .16→.44, fixtures .05→.455, table rows .0→.57
- [x] Hover transitions: rows .15s, crest .3s, buttons/chips .2s, bars .2s
- [x] Scroll reveal `.rv` → `.in` with IntersectionObserver, rootMargin -8%
- [x] Count-up 900ms ease-out for percentages and season numbers
- [x] `prefers-reduced-motion:reduce` disables all animation/transition/smooth scroll
- [x] Reduced motion also skips count-up (numbers render final immediately)

## 10. Breakpoints

- [x] ≤1100px: `.grid` becomes one column (side cards drop below fixtures)
- [x] ≤1023px: sidebar off-canvas, main margin 0 / padding 20, menu visible
- [x] ≤700px: fixture row areas `time tag / home / bar / away`; home row reversed;
      match header/chance cards/g2 single column; hero padding 22, slash hidden; main padding 16
- [x] 390px: single column, stacked fixture rows, no overflow

## 11. Copy (exact)

- [x] "Skip to content", "EPL Intelligence", "Football", "Premier League", "Overview",
      "Matchweek 11", "Table", "This weekend", "10 fixtures · Saturday 21 Nov · 15:00",
      "Results up to 8 November 2026", "☰ Menu"
- [x] Footer sentence 1: "Chances are the project's own match ratings for 2026/27 — a read on the fixture, not betting odds and not a guarantee."
- [x] Footer sentence 2: "Form, table and head-to-head use real Premier League results up to 8 November 2026."
- [x] Week note: "Chances are rounded to whole percentages and always add up to 100."
- [x] Empty filter: "No fixtures lean this way this week."
- [x] "Win in 90 minutes", "Points shared", "Our pick"
- [x] Table key: "Top four", "Bottom three"
- [x] Drawer: "Last five (oldest → newest)", "Home and away", "Home points", "Away points"

## 12. Data

- [x] 10 fixtures with the exact percentages; each row sums to 100
- [x] 20 table rows with exact P/W/D/L/GF/GA/GD/Pts and last-five strings
- [x] Season numbers: 100 matches, 273 goals, 2.73/game, 42 H wins, 27 draws, 31 A wins
- [x] Only Liverpool v Man United has rates/head-to-head; others "—"
- [x] Badge map: 19 ids, Coventry fallback; HUL 88, IPS 40, ARS 3, LIV 14, MUN 1
- [x] Logo load failure replaces the `<img>` with the built-in SVG badge
- [x] No number hard-coded in markup; all from `data.py`

## 13. Rules

- [x] No banned words anywhere in rendered UI: model, algorithm, prediction, probability,
      accuracy, confidence, calibration, features, training data, dataset, machine learning,
      AI, neural, data-driven
- [x] No accuracy/performance claims, no odds, no betting UI
- [x] Missing values show "—" with `<!-- TODO: real value -->`
- [x] Footer sentences unchanged

## 14. Accessibility

- [x] Skip link first in DOM
- [x] `main` has `tabindex="-1"`
- [x] Sidebar `aria-label="Main navigation"`, nav `aria-label="Sections"`
- [x] Menu button `aria-controls`/`aria-expanded`
- [x] Drawer `role="dialog" aria-modal="true" aria-label="Team summary"`, X `aria-label="Close"`
- [x] Bars have descriptive `role="img"` labels
- [x] Form chips have `aria-label="Last five: …"`
- [x] Table rows `role="button"` + Enter opens drawer
- [x] Escape closes drawer and menu
- [x] Focus visible everywhere

## 15. Network / runtime

- [x] Only requests: Google Fonts + `resources.premierleague.com` badge images
- [x] No console errors
- [x] Works at 1440, 1024, 390 with no horizontal overflow
- [x] `streamlit run app.py` from a clean install with `requirements.txt`
