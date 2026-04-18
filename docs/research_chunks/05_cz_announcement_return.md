### Earnings Announcement Return (EAR / AnnouncementReturn)

### Origin and Theoretical Motivation

The Earnings Announcement Return signal traces to Brandt, Kishore, Santa-Clara, and Venkatachalam (2008), *"Earnings Announcements are Full of Surprises"* (SSRN 909563). The authors argue that the price reaction in a tight window around the earnings release is a sufficient statistic for the *total* information content of the announcement — not only the headline EPS surprise versus consensus (SUE), but also unexpected information about sales, margins, guidance, capex, and the qualitative tone of the call. They show that a long-short portfolio sorted on EAR earns a 7.55% annual abnormal return, exceeding a SUE-sorted strategy by 1.3 ppt and remaining largely orthogonal to it. The signal is intellectually adjacent to Frazzini and Lamont (2007) *"The Earnings Announcement Premium and Trading Volume"*, who document a related event-window premium.

### Construction

EAR is the cumulative return on a 3-day (sometimes 4-day) window centered on the announcement date `t`:

- Pre-market / before-open releases: `[t-1, t+1]`
- After-market-close (AMC) releases: `[t, t+2]` — because the price impact lands in the next trading session

The sign is **positive**: large positive announcement returns predict continued outperformance over the following 30–60 trading days. The economic content is a generalised post-earnings-announcement-drift (PEAD) — markets fail to fully impound event-window information, leaving a slow diffusion that arbitrageurs can harvest.

### Empirical Performance in Our Stack

Per `cz_signal_ic.csv`, `AnnouncementReturn` produces an **IC information ratio of 0.254**, our 3rd-highest novel signal in the Chen-Zimmermann research sweep — strong enough to justify inclusion despite EAR's relatively narrow universe (event-conditional).

### Implementation Notes

- Code: `src/alt_features.py:earnings_ann_return_signal` (lines 608–656).
- Window logic (Phase D fix, April 2026): the `before_after_market` field returned by EODHD selects the window — `"amc"` triggers `[t, t+2]`, otherwise the symmetric `[-1, +1]` window applies.
- **Latent bug discovered 2026-04-17**: the EODHD response field is snake_case `before_after_market`, but earlier parser code in `src/alt_data_loader.py:load_earnings_calendar_eodhd` had silently defaulted every record to `"unknown"`, collapsing the AMC branch and degrading the signal to the symmetric default. Fixed at lines 1005–1013, with an automatic cache-migration probe (lines 895–927) that detects pre-fix caches and forces one-time re-fetch.
- Data dependency: EODHD bulk earnings calendar (`/api/calendar/earnings`), 281 tickers cached, 2013 → 2026 (`load_earnings_calendar_eodhd`).
- Carry: signal value is held forward for `surprise_carry_days` (currently 5–10 trading days, tuned in Phase D) — matching the empirical PEAD half-life from Bernard-Thomas (1989) and the Brandt et al. drift window.

### Sources

- [Earnings Announcements are Full of Surprises — SSRN 909563](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=909563)
- [Quantpedia mirror PDF](https://quantpedia.com/www/Earnings_Announcements_are_Full_of_Surprises.pdf)
- [Semantic Scholar entry](https://www.semanticscholar.org/paper/Earnings-Announcements-are-Full-of-Surprises-Kishore-Brandt/3cb316e8f28bc359cbb92fe985c9011a2998198f)
