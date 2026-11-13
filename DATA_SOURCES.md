# Data Sources

Every number in this project comes from a public, citable source. Raw payloads are
cached byte-for-byte in `data/raw/` and never edited. If a source is unavailable,
the pipeline degrades to the cached copy and records the failure in
`data/processed/update_log.json` — it never invents rows.

## 1. football-data.co.uk — primary source

| Property | Value |
|---|---|
| Season results | `https://www.football-data.co.uk/mmz4281/{code}/E0.csv` |
| Upcoming fixtures | `https://www.football-data.co.uk/fixtures.csv` |
| Format | CSV, `utf-8-sig`, comma separated |
| Licence | Free for personal use; see the site's terms |
| Cached as | `data/raw/E0_{season_code}.csv`, `data/raw/fixtures.csv` |

`E0.csv` is the Premier League results file. The project uses:

- `Date`, `Time`, `HomeTeam`, `AwayTeam`
- `FTHG`, `FTAG`, `FTR` (full-time goals and result)
- `HS`, `AS`, `HST`, `AST`, `HC`, `AC`, `HF`, `AF`, `HY`, `AY`, `HR`, `AR`
  (shots, shots on target, corners, fouls, cards)
- `B365*`, `BW*`, `IW*`, `PS*`, `WH*`, `VC*`, `BFD*` bookmaker prices, plus the
  `Avg*` and `Max*` aggregate series

### Season codes

`SEASONS` in `src/config.py` maps the compact code used by the source to a
readable label. **2020/21 is the one irregular code (`2021`)**; naively
incrementing the year produces a 404 for that season.

| Code | Season | Code | Season |
|---|---|---|---|
| 1415 | 2014/15 | 2021 | 2020/21 |
| 1516 | 2015/16 | 2122 | 2021/22 |
| 1617 | 2016/17 | 2223 | 2022/23 |
| 1718 | 2017/18 | 2324 | 2023/24 |
| 1819 | 2018/19 | 2425 | 2024/25 |
| 1920 | 2019/20 | 2526 | 2025/26 |
| | | 2627 | 2026/27 |

Thirteen seasons were downloaded: 2014/15 → 2026/27 (the current season is
partial). That yields **4,660 completed matches**, 36 distinct clubs, and a date
range of **2014-08-16 → 2026-11-08**.

### Derived market fields

Each price source is converted to implied probabilities by removing its own
overround:

```
raw_i     = 1 / decimal_odds_i
overround = sum(raw_i)
p_i       = raw_i / overround
```

The consensus `mkt_home/draw/away` is the **row-wise mean of every available
source**: seven bookmaker triples plus the average-odds and maximum-odds series.
`mkt_books` counts the number of contributing **price sources** (4–8, median 6),
not distinct bookmakers, and `mkt_disp_*` is the standard deviation across those
sources. The consensus is verified by reconstruction in
`reports/research/audit.json` (maximum absolute error 0.0).

These are the `mkt_*` columns. They are **pre-match** by definition (prices are
published before kick-off) but they are excluded from the `no_odds` model
variant. The source does not publish a precise timestamp per price, so the
project treats the snapshot as closing consensus; this is a documented
approximation, not a verified publication time.

## 2. openfootball / football.json — fixture fallback

| Property | Value |
|---|---|
| URL | `https://raw.githubusercontent.com/openfootball/football.json/master/2026-27/en.1.json` |
| Format | JSON |
| Cached as | `data/raw/openfootball_2026-27.json` |
| Role | Supplies the complete 2026/27 schedule and round numbers when the football-data fixture feed has no future EPL rows |

The fixture resolution order in `src/data/processing.load_fixtures` is:

1. Future rows in `fixtures.csv` whose teams are known and whose date is after
   the last completed match.
2. Otherwise, the openfootball schedule for dates after the data cutoff.
3. Otherwise, an empty frame — the dashboard shows an explicit empty state
   rather than a fabricated fixture list.

## 3. What is *not* used

- No scraped social media, injury news, or transfer data: none of it can be
  reconstructed historically without look-ahead bias.
- No understat / FBref expected-goals feeds: they are not available on a
  consistent 13-season horizon through a stable public interface, and mixing
  providers would create coverage gaps.
- No synthetic or interpolated matches. Missing statistics remain `NaN` and are
  handled by the model's imputer.

## 4. Freshness and reproducibility

| Artefact | Location | Refresh |
|---|---|---|
| Raw season CSVs | `data/raw/` | `python -m src.data.acquisition` |
| Processed match table | `data/processed/matches.csv` | `python -m src.pipeline` |
| Upcoming fixtures | `data/fixtures/upcoming.csv` | `python -m src.pipeline` |
| Update audit log | `data/processed/update_log.json` | appended on every refresh |
| Quality report | `reports/data_quality/data_quality.json` | `python -m src.pipeline` |
| Leakage audit | `reports/data_quality/leakage_report.json` | `python -m src.data.validation` |

The GitHub Actions workflow `.github/workflows/update.yml` runs the same code
daily at 06:15 UTC and commits the refreshed tables and predictions, so the
repository always carries a timestamped record of what the model saw.

## 5. Known limitations

- football-data prices are a *consensus snapshot*, not a timestamped order book.
  The feed does not state a publication time, so the project cannot prove that
  every price was available at a precise pre-match instant; it is treated as a
  closing snapshot throughout.
- Early seasons have fewer price sources; `mkt_books` exposes the count (4–8,
  median 6) so the model can learn to trust thin markets less.
- `mkt_books` counts price sources (bookmakers plus the average and maximum
  series), not distinct bookmakers.
- The 2026/27 season is in progress; metrics that include it are labelled
  "partial" and are never used for model selection.
- Team names are normalised through `TEAM_ALIASES`; a genuinely new club would
  need an alias entry before it can be processed.
