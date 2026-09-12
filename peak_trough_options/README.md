# Peak and trough forecasting for option selection

Forecasts **how far a stock travels up and down over the life of an option, and
when** — then turns that band into ranked option structures.

> **This is educational research code, not investment advice.** It produces
> model estimates from historical prices. Options routinely expire worthless
> and can lose 100% of their premium. Read [Limitations](#limitations-read-this-part)
> before you let any of this influence a real trade.

---

## Why excursions instead of direction

Most price models predict a direction, or the return at some fixed date. Neither
is the question an options buyer actually faces.

A call bought today does not care where the stock closes on average. It cares
about the **highest price reached before expiry** — that sets the strike that
can pay, and whether there was ever a moment worth closing into. A put's value
depends on the **lowest** price. A straddle needs the *width* of the range.
Expiry choice depends on **when** the extreme lands.

So this module predicts the conditional distribution of three quantities over
the next `H` trading days:

| Target | Meaning | What it decides |
| --- | --- | --- |
| `mfe` | maximum favourable excursion: `max(high) / close - 1` | call strikes, profit targets |
| `mae` | maximum adverse excursion: `min(low) / close - 1` | put strikes, stops, short-put safety |
| `ret` | terminal return: `close[t+H] / close[t] - 1` | what a European payoff settles against |

plus `t_peak` and `t_trough`, the number of bars until each extreme — which is
how you decide whether a 30-day contract has enough time.

Nothing here produces a point forecast. For strike selection the shape of the
tail *is* the question, and a conditional mean tells you nothing about how much
room a contract needs.

---

## Quick start

```bash
# No network or data needed — synthetic bars, end to end
python -m peak_trough_options.cli --demo --validate

# Your own OHLCV csv, with the implied vol from the real option chain
python -m peak_trough_options.cli --csv AAPL.csv --horizon 30 --iv 0.32 --validate

# Pull daily bars from stooq (needs pandas-datareader)
python -m peak_trough_options.cli --ticker AAPL.US --horizon 21 --iv 0.28
```

As a library:

```python
from peak_trough_options import load_csv, run

bars = load_csv('AAPL.csv')                       # date, open, high, low, close[, volume]
view = run(bars, horizon=21, implied_sigma_annual=0.32)

print(view['targets'])            # peak / trough / expiry price at each quantile
print(view['timing'])             # median bars to the peak and the trough
print(view['volatility_edge'])    # model width vs the market's implied width
print(view['structures'].head())  # ranked structures with cost, EV and greeks
```

**Always pass `--iv` / `implied_sigma_annual` from the real option chain.**
Without it the code substitutes `realised vol × 1.10`, and every expected-profit
number then inherits that guess. The CLI marks the output `[PROXY]` when it does.

---

## How it works

```
 bars ──► features.py ────┐
   │      (causal only)   ├──► model.py ──► quantile bands ──► options.py ──► ranked
   └────► labels.py ──────┘    (GBM per       (mfe/mae/ret)     (BS pricing)   structures
          (MFE/MAE/ret)         quantile)
                                    ▲
                          validation.py — purged walk-forward
```

**1. Volatility scaling.** Raw excursions are not stationary: a 6% move means
something different in a calm month than a panicked one. Targets are divided by
`sigma_t · sqrt(H)`, where `sigma_t` is the EWMA volatility known *at the
decision bar*. Models learn in that scaled space and predictions are multiplied
back out. This is what lets one model span volatility regimes.

**2. Quantile gradient boosting.** One `HistGradientBoostingRegressor` with
pinball loss per (target, quantile) pair. Independently fitted quantiles can
cross; `monotone_rearrange` sorts them, which is the Chernozhukov–Fernández-Val–Galichon
rearrangement and provably never increases the loss.

**3. Swing structure.** An ATR-scaled zig-zag marks confirmed peaks and troughs.
A pivot is only *confirmed* once price has retraced `k · ATR` away from it, so
the bar the extreme happened on and the bar you could have known about it are
different. Features reference the confirmation bar. Getting this backwards is
one of the most common ways a swing-trading backtest fools its author.

**4. Conformal calibration.** Boosted quantiles fitted on overlapping labels
come out reliably overconfident — measured here at 68–72% coverage for a
nominal 80% interval. A purged tail of the training window is held out, and
each quantile is shifted by the offset that makes it hit its nominal level on
that block (split-conformal quantile regression, Romano et al. 2019). On 2,800
synthetic bars this cut the worst coverage error from 0.18 to 0.08. Below
roughly `8 × H` calibration rows the correction is noisier than the error it
fixes, so it is skipped rather than applied badly — check `forecaster.calibrated_`.

**5. Distribution from quantiles.** `QuantileDistribution` interpolates in
*normal-score space* rather than probability space, so five interior quantiles
extrapolate to sensible tails instead of flattening into a point mass. It
reproduces a Gaussian exactly from any two of its quantiles.

**6. Option scoring.** Premium is priced with Black–Scholes at the market's
implied vol; the payoff is integrated under the *model's* distribution. The gap
between the two is the edge. Structures are ranked by expected profit per unit
of capital at risk, after a bid-ask haircut that always works against you.

---

## Guarding against lookahead

A peak/trough model is unusually easy to fool yourself with, because the labels
are defined by the future and the features are tempting to centre. Three
independent defences, all enforced by tests:

**Causal features.** Every feature is recomputed on a truncated copy of the
history; if any value at bar `t` changes when the bars after `t` are deleted,
the test fails. This catches the whole class of bugs at once.

**Purged, embargoed walk-forward.** A label at bar `t` resolves at `t+H`, so
training rows in `[test_start − H, test_start)` know the answer. Those rows are
dropped, plus an embargo. `test_validation.py` asserts the gap is exactly
`H + embargo` on every fold.

**Canaries.** With labels randomly permuted the model must not beat climatology
(`skill < 0.05`). On a pure random walk it must not show large skill.

---

## Reading the output

The benchmark everywhere is **climatology**: the unconditional quantiles of the
scaled excursion, measured on the training window alone. That baseline already
knows the typical size of a move for this asset, so beating it is the only
evidence the features say something about *this particular setup* rather than
about volatility in general.

```
target  level  model_pinball  baseline_pinball   skill  coverage
   mfe 0.9000         0.1782            0.1811  0.0161    0.9046
```

* **`skill`** — fractional reduction in pinball loss versus climatology.
  Positive means the conditioning information helped. **Negative means it did
  not, and you should trade the climatology band instead of the model.**
* **`coverage`** — realised `P(outcome ≤ predicted quantile)`. Should land on
  the nominal level. If the 10% quantile covers 30%, the band is too high.
* **`model_width` vs `baseline_width`** — a model much narrower than
  climatology while under-covering is overconfident, not skilful.

Expect skill in the low single-digit percents when it is real. Anything above
~0.2 on daily equity data should be treated as a bug until proven otherwise.

`--validate` ends with a one-line verdict (`backtest.verdict`) that says which
of those cases you are in:

```
Verdict: No edge: the model does not beat training-window climatology.
         Use the climatology band, not this model.
  mean skill -0.114
```

That is the honest outcome on the synthetic demo data, and it is a common
outcome on real tickers. The code says so rather than burying it.

### Effective sample size

This is the thing that bites hardest. With `H = 21`, consecutive labels overlap
by 20 days: 1,500 training rows carry roughly `1500 / 21 ≈ 70` independent
observations. The default hyperparameters are deliberately heavily regularised
for that reality. If you loosen them, watch the interval coverage collapse.

Practical minimum: about **2,500 daily bars (10 years)** for a 21-day horizon.
Below roughly 1,500 the conformal calibration switches itself off and the bands
revert to being too narrow. `fit` refuses outright below 250 labelled rows.

---

## Module map

| File | Contents |
| --- | --- |
| `data.py` | CSV / stooq loading, column normalisation, bar validation, synthetic generator |
| `swings.py` | Wilder ATR, ATR-scaled zig-zag pivots with confirmation lag, swing features |
| `features.py` | ~50 causal features: momentum, volatility estimators, range position, variance ratios, trend quality, oscillators, volume |
| `labels.py` | Forward extremes and their volatility-scaled forms |
| `validation.py` | Purged, embargoed walk-forward splitter; decay weights |
| `distribution.py` | Monotone rearrangement, pinball loss, `QuantileDistribution`, closed-form running-max quantiles |
| `model.py` | `PeakTroughForecaster` — quantile GBM per target and level, with split-conformal calibration |
| `options.py` | Black–Scholes with dividends, greeks, implied vol, nine structures, EV scoring, ranking |
| `backtest.py` | Walk-forward evaluation, calibration tables, plain-language `verdict`, decision replay, volatility sensitivity sweep |
| `pipeline.py` | `prepare` → `fit` → `latest_forecast` → `recommend` |
| `cli.py` | Command line front end |

---

## Limitations (read this part)

**No option chain.** The module never sees a real quote. It prices with
Black–Scholes at a single implied vol you supply. There is no volatility smile,
no term structure, no American early exercise, no borrow cost, no assignment
risk, and no check that the strikes it names are actually listed or liquid.
Real spreads on illiquid strikes will exceed the modelled haircut, often by a
lot.

**`simulate_decisions` is not a P&L backtest.** With no chain, it prices
premium at `realised vol × vrp_multiple`. It tests whether the forecast leads to
sensible structure choices under a pricing assumption — it does *not* establish
profitability. `vrp_sensitivity` sweeps that assumption on purpose: **if the
sign of the result flips across the sweep, the simulation is measuring your
volatility guess, not the forecast.** It usually does flip.

**Earnings and events.** Daily OHLC carries no earnings calendar. Excursions
around a scheduled event are drawn from a different distribution than the model
was fitted on, and implied vol is elevated for a reason the model cannot see.
Do not use this across an earnings date without handling that yourself.

**Survivorship and corporate actions.** Feed it split- and dividend-adjusted
prices. Unadjusted series produce fake gaps that the swing detector will read as
genuine reversals.

**Regime change.** Fitted on history, applied to the future. Every calibration
number above is measured on the past. The 2020 volatility spike was outside any
band a 2019-trained model would have drawn.

**Multiple comparisons.** Nine structures are ranked on every call. The best of
nine noisy estimates is biased upward. Treat the ranking as a shortlist to think
about, not a verdict.

**It may simply have no edge on your data.** Negative skill is a normal and
frequent outcome, and the code reports it honestly rather than hiding it. That
is the intended behaviour, not a bug to tune away.

---

## Tests

```bash
python -m pytest peak_trough_options/tests -q
```

The suite covers the causality guarantees, label correctness against a naive
reference implementation, splitter purging, Black–Scholes against put-call
parity and finite-difference greeks, the reflection-principle formula against
Monte Carlo, and end-to-end calibration.
