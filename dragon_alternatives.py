#!/usr/bin/env python3
"""
Dragon Portfolio — Alternativas
Objetivo: Doblar el Sharpe Ratio del Dragon Portfolio base.

Estrategias:
  A) Dual Momentum — filtro absoluto (mom<0 → SHY/cash)
  B) SMA200 Trend Filter — risk assets debajo de SMA200 → reducir exposicion
  C) Risk Parity — pesos por inversa de volatilidad (60d rolling)
  D) Dual Momentum + Trend — Combina A + B
  E) Optimized Levered — Mejor estrategia con leverage 1.5x @ 4% costo deuda

SFinance-alicIA
"""
import yfinance as yf
import numpy as np
import datetime, os, math, json

# ═══════════════════════════════════════════════════════════════════
# 1. CONFIG (same universe as base Dragon)
# ═══════════════════════════════════════════════════════════════════
UNIVERSES = {
    "Equity":      ["SPY", "QQQ", "IWM", "EEM", "VGK", "EWY", "EWP", "EWZ", "EPOL"],
    "Bonds":       ["SHY", "IEF", "TLT", "TIP", "LQD"],
    "HardAssets":  ["GLD", "SLV", "CPER", "BTC-USD"],
    "LongVol":     ["BTAL"],
    "Commodities": ["DBC"],
}
ALL_TICKERS = sorted(set(t for lst in UNIVERSES.values() for t in lst))
LATE_JOINERS = {"BTC-USD"}
CORE_TICKERS = sorted(t for t in ALL_TICKERS if t not in LATE_JOINERS)

N_SELECT = 3
MOM_LOOKBACK = 126
SMA_LONG = 200        # SMA200 for trend filter
SMA_CMDTY = 50
VOL_LOOKBACK = 60     # Rolling vol window for risk parity
ABS_MOM_THRESH = 0.0  # Absolute momentum threshold

START = "2006-03-01"
TODAY = datetime.date.today()
RF_ANNUAL = 0.043
BTAL_LEVERAGE = 1.0

# Leverage config
MAX_LEVERAGE = 1.5
DEBT_COST_ANNUAL = 0.04
DEBT_COST_DAILY = DEBT_COST_ANNUAL / 252

# Base Dragon weights
W_DRAGON = {"Equity": 0.24, "Bonds": 0.18, "HardAssets": 0.19, "LongVol": 0.21, "CmdtyTrend": 0.18}

COLORS = {
    "Equity": "#10b981", "Bonds": "#06b6d4", "HardAssets": "#f59e0b",
    "LongVol": "#ef4444", "CmdtyTrend": "#a855f7",
    "Base":      "#94a3b8",
    "DualMom":   "#f59e0b",
    "SMA200":    "#06b6d4",
    "RiskParity":"#a855f7",
    "DualTrend": "#10b981",
    "Levered":   "#ef4444",
    "6040":      "#475569",
    "SPY":       "#3b82f6",
}

STRAT_NAMES = {
    "Base":      "Dragon Base",
    "DualMom":   "A) Dual Momentum",
    "SMA200":    "B) SMA200 Trend",
    "RiskParity":"C) Risk Parity",
    "DualTrend": "D) Dual Mom+Trend",
    "Levered":   "E) Optimized Lev 1.5x",
}

TICKER_LABELS = {
    "SPY": "US Large", "QQQ": "Nasdaq", "IWM": "Small Cap",
    "EEM": "Emergentes", "VGK": "Europa", "EWY": "Korea",
    "EWP": "Espana", "EWZ": "Brasil", "EPOL": "Polonia",
    "SHY": "1-3Y", "IEF": "7-10Y", "TLT": "20+Y", "TIP": "TIPS", "LQD": "IG Corp",
    "GLD": "Oro", "SLV": "Plata", "CPER": "Cobre", "BTC-USD": "Bitcoin",
    "BTAL": "Anti-Beta", "DBC": "Commodities",
}

# ═══════════════════════════════════════════════════════════════════
# 2. DATA FETCHING (reuse cache from base)
# ═══════════════════════════════════════════════════════════════════
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "price_cache.json")

print("═══ Dragon Portfolio — ALTERNATIVAS ═══\n")

def load_cache():
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
        if cache.get("date") == str(TODAY):
            print("Using cached data (same day)...")
            return cache
    except:
        pass
    return None

def save_cache(dates_list, data_dict):
    cache = {
        "date": str(TODAY),
        "dates": [str(d) for d in dates_list],
        "prices": {t: [None if (isinstance(v, float) and np.isnan(v)) else v
                       for v in data_dict[t].tolist()] for t in data_dict},
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)
    print(f"  Cache saved: {CACHE_FILE}")

cached = load_cache()

if cached:
    dates = [datetime.date.fromisoformat(d) for d in cached["dates"]]
    N = len(dates)
    price_data = {}
    for t in cached["prices"]:
        if t in ALL_TICKERS:
            price_data[t] = np.array([np.nan if v is None else v for v in cached["prices"][t]])
    for t in price_data:
        valid = np.count_nonzero(~np.isnan(price_data[t]))
        print(f"  + {t}: {valid}/{len(price_data[t])} days (cached)")
    for t in ALL_TICKERS:
        if t not in price_data:
            print(f"  ! {t} missing from cache — forcing re-download")
            cached = None
            break

if not cached:
    print("Fetching data from Yahoo Finance...")
    prices = {}
    for ticker in ALL_TICKERS:
        df = yf.download(ticker, start=START, end=str(TODAY), progress=False, auto_adjust=True)
        if len(df) > 100:
            prices[ticker] = df[["Close"]].copy()
            prices[ticker].columns = [ticker]
            print(f"  + {ticker}: {len(df)} days")
        else:
            print(f"  x {ticker}: insufficient data ({len(df)} days)")

    common_idx = prices[CORE_TICKERS[0]].index
    for t in CORE_TICKERS[1:]:
        common_idx = common_idx.intersection(prices[t].index)
    common_idx = common_idx.sort_values()

    price_data = {}
    for t in CORE_TICKERS:
        price_data[t] = prices[t].loc[common_idx, t].values.astype(float)
    for t in LATE_JOINERS:
        if t in prices:
            merged = prices[t].reindex(common_idx)
            price_data[t] = merged[t].values.astype(float)
        else:
            price_data[t] = np.full(len(common_idx), np.nan)

    dates = [d.date() if hasattr(d, "date") else d for d in common_idx]
    N = len(dates)
    save_cache(dates, price_data)

print(f"\n  Aligned: {N} trading days ({dates[0]} -> {dates[-1]})")

# Daily returns
ret = {}
for t in ALL_TICKERS:
    p = price_data[t]
    ret[t] = np.diff(p) / p[:-1]
dates_ret = dates[1:]
N_ret = len(dates_ret)

# ═══════════════════════════════════════════════════════════════════
# 3. SIGNALS: Momentum, SMA200, Rolling Vol
# ═══════════════════════════════════════════════════════════════════
print("\nComputing signals...")

# 3a. Momentum (6-month)
mom = {}
for t in ALL_TICKERS:
    p = price_data[t]
    m = np.full(N_ret, np.nan)
    for i in range(N_ret):
        if i >= MOM_LOOKBACK:
            p_now = p[i + 1]  # price index is shifted by 1 vs ret index
            p_prev = p[i + 1 - MOM_LOOKBACK]
            if not np.isnan(p_now) and not np.isnan(p_prev) and p_prev > 0:
                m[i] = p_now / p_prev - 1
    mom[t] = m

# 3b. SMA200 trend signal (True = above SMA200)
sma200_above = {}
for t in ALL_TICKERS:
    p = price_data[t]
    signal = np.full(N, False)
    for i in range(N):
        if i >= SMA_LONG:
            window = p[i - SMA_LONG + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= SMA_LONG * 0.8:
                sma_val = np.mean(valid)
                signal[i] = p[i] > sma_val
    sma200_above[t] = signal[1:]  # align with returns

# 3c. Rolling volatility (annualized, 60-day)
rolling_vol = {}
for t in ALL_TICKERS:
    r = ret[t]
    rv = np.full(N_ret, np.nan)
    for i in range(VOL_LOOKBACK, N_ret):
        window = r[i - VOL_LOOKBACK:i]
        valid = window[~np.isnan(window)]
        if len(valid) >= VOL_LOOKBACK * 0.7:
            rv[i] = np.std(valid) * np.sqrt(252)
    rolling_vol[t] = rv

print("  + Momentum 126d, SMA200 trend, Rolling vol 60d: OK")

# ═══════════════════════════════════════════════════════════════════
# 4. SELECTION LOGIC PER STRATEGY
# ═══════════════════════════════════════════════════════════════════
print("\nRunning selection logic...")

# --- STRATEGY BASE: Same as original ---
def run_base_selection():
    selections = {block: [] for block in UNIVERSES}
    current_sel = {}
    for block, candidates in UNIVERSES.items():
        ns = min(N_SELECT, len(candidates))
        defaults = [t for t in candidates if t not in LATE_JOINERS][:ns]
        if len(defaults) < ns:
            defaults = candidates[:ns]
        current_sel[block] = defaults

    for i in range(N_ret):
        d = dates_ret[i]
        is_rebal = (i == 0) or (d.month != dates_ret[i - 1].month)
        if is_rebal:
            for block, candidates in UNIVERSES.items():
                ns = min(N_SELECT, len(candidates))
                valid_c = [(t, mom[t][i]) for t in candidates if not np.isnan(mom[t][i])]
                valid_c.sort(key=lambda x: -x[1])
                if len(valid_c) >= ns:
                    current_sel[block] = [t for t, _ in valid_c[:ns]]
                elif len(valid_c) > 0:
                    current_sel[block] = [t for t, _ in valid_c]
        for block in UNIVERSES:
            selections[block].append(list(current_sel[block]))
    return selections

# --- STRATEGY A: Dual Momentum (absolute mom filter → SHY) ---
def run_dual_mom_selection():
    """Same top-3 but if a pick has negative absolute momentum, replace with SHY."""
    selections = {block: [] for block in UNIVERSES}
    current_sel = {}
    for block, candidates in UNIVERSES.items():
        ns = min(N_SELECT, len(candidates))
        defaults = [t for t in candidates if t not in LATE_JOINERS][:ns]
        if len(defaults) < ns:
            defaults = candidates[:ns]
        current_sel[block] = defaults

    for i in range(N_ret):
        d = dates_ret[i]
        is_rebal = (i == 0) or (d.month != dates_ret[i - 1].month)
        if is_rebal:
            for block, candidates in UNIVERSES.items():
                ns = min(N_SELECT, len(candidates))
                valid_c = [(t, mom[t][i]) for t in candidates if not np.isnan(mom[t][i])]
                valid_c.sort(key=lambda x: -x[1])
                if len(valid_c) >= ns:
                    picks = [t for t, _ in valid_c[:ns]]
                elif len(valid_c) > 0:
                    picks = [t for t, _ in valid_c]
                else:
                    picks = list(current_sel[block])
                # Absolute momentum filter: replace negative-mom picks with SHY
                filtered = []
                for t in picks:
                    if not np.isnan(mom[t][i]) and mom[t][i] < ABS_MOM_THRESH:
                        filtered.append("SHY")  # cash proxy
                    else:
                        filtered.append(t)
                current_sel[block] = filtered
        for block in UNIVERSES:
            selections[block].append(list(current_sel[block]))
    return selections

# --- STRATEGY B: SMA200 Trend Filter ---
def run_sma200_selection():
    """Same top-3 but scale down exposure if below SMA200."""
    selections = {block: [] for block in UNIVERSES}
    exposure_scale = {block: [] for block in UNIVERSES}  # fraction in risk vs cash
    current_sel = {}
    for block, candidates in UNIVERSES.items():
        ns = min(N_SELECT, len(candidates))
        defaults = [t for t in candidates if t not in LATE_JOINERS][:ns]
        if len(defaults) < ns:
            defaults = candidates[:ns]
        current_sel[block] = defaults

    for i in range(N_ret):
        d = dates_ret[i]
        is_rebal = (i == 0) or (d.month != dates_ret[i - 1].month)
        if is_rebal:
            for block, candidates in UNIVERSES.items():
                ns = min(N_SELECT, len(candidates))
                valid_c = [(t, mom[t][i]) for t in candidates if not np.isnan(mom[t][i])]
                valid_c.sort(key=lambda x: -x[1])
                if len(valid_c) >= ns:
                    current_sel[block] = [t for t, _ in valid_c[:ns]]
                elif len(valid_c) > 0:
                    current_sel[block] = [t for t, _ in valid_c]

        # Compute how many picks are above SMA200
        picks = current_sel[block]
        above_count = sum(1 for t in picks if t in sma200_above and sma200_above[t][i])
        scale = above_count / max(len(picks), 1)  # 0 to 1
        # Minimum 30% exposure even when all below SMA
        scale = max(scale, 0.3)

        for block_name in UNIVERSES:
            selections[block_name].append(list(current_sel[block_name]))
        # We handle the scale in the return calculation
        for block_name in UNIVERSES:
            if len(exposure_scale[block_name]) < i + 1:
                picks_b = current_sel[block_name]
                above_b = sum(1 for t in picks_b if t in sma200_above and sma200_above[t][i])
                sc = above_b / max(len(picks_b), 1)
                sc = max(sc, 0.3)
                exposure_scale[block_name].append(sc)

    # Fix: properly build exposure_scale for all days
    # Redo this cleanly
    selections2 = {block: [] for block in UNIVERSES}
    exposure2 = {block: [] for block in UNIVERSES}
    current_sel2 = {}
    for block, candidates in UNIVERSES.items():
        ns = min(N_SELECT, len(candidates))
        defaults = [t for t in candidates if t not in LATE_JOINERS][:ns]
        if len(defaults) < ns:
            defaults = candidates[:ns]
        current_sel2[block] = defaults

    for i in range(N_ret):
        d = dates_ret[i]
        is_rebal = (i == 0) or (d.month != dates_ret[i - 1].month)
        if is_rebal:
            for block, candidates in UNIVERSES.items():
                ns = min(N_SELECT, len(candidates))
                valid_c = [(t, mom[t][i]) for t in candidates if not np.isnan(mom[t][i])]
                valid_c.sort(key=lambda x: -x[1])
                if len(valid_c) >= ns:
                    current_sel2[block] = [t for t, _ in valid_c[:ns]]
                elif len(valid_c) > 0:
                    current_sel2[block] = [t for t, _ in valid_c]
        for block in UNIVERSES:
            selections2[block].append(list(current_sel2[block]))
            picks_b = current_sel2[block]
            above_b = sum(1 for t in picks_b if t in sma200_above and sma200_above[t][i])
            sc = above_b / max(len(picks_b), 1)
            sc = max(sc, 0.3)
            exposure2[block].append(sc)

    return selections2, exposure2

# ═══════════════════════════════════════════════════════════════════
# 5. RETURN CONSTRUCTION FOR EACH STRATEGY
# ═══════════════════════════════════════════════════════════════════

# Helper: block returns from selections
def block_returns(selections, block_name, exposure_scale=None):
    """Equal-weight average of selected tickers, with optional exposure scaling."""
    r = np.zeros(N_ret)
    shy_ret = ret["SHY"]
    for i in range(N_ret):
        picks = selections[block_name][i]
        valid = []
        for t in picks:
            rv = ret[t][i] if t in ret and not np.isnan(ret[t][i]) else 0.0
            valid.append(rv)
        if valid:
            r[i] = np.mean(valid)
        if exposure_scale is not None:
            # Scale: exposure_scale portion in risk, rest in SHY
            sc = exposure_scale[block_name][i]
            shy_r = shy_ret[i] if not np.isnan(shy_ret[i]) else 0.0
            r[i] = sc * r[i] + (1 - sc) * shy_r
    return r

def longvol_returns():
    return ret["BTAL"] * BTAL_LEVERAGE

def cmdty_trend_returns():
    dbc_prices = price_data["DBC"]
    r = np.zeros(N_ret)
    shy_ret = ret["SHY"]
    for i in range(N_ret):
        day_idx = i + 1
        if day_idx >= SMA_CMDTY:
            sma50 = np.mean(dbc_prices[day_idx - SMA_CMDTY:day_idx])
            deviation = (dbc_prices[day_idx] / sma50) - 1
            if deviation > 0:
                weight = min(deviation / 0.05, 1.0)
                r[i] = ret["DBC"][i] * weight
            else:
                r[i] = 0.0
        else:
            r[i] = ret["DBC"][i] * 0.5
    return r

def cmdty_trend_sma200():
    """Commodity trend with additional SMA200 filter."""
    dbc_prices = price_data["DBC"]
    r = np.zeros(N_ret)
    for i in range(N_ret):
        day_idx = i + 1
        # SMA200 gate
        if day_idx >= SMA_LONG:
            sma200 = np.mean(dbc_prices[day_idx - SMA_LONG:day_idx])
            if dbc_prices[day_idx] < sma200:
                r[i] = 0.0
                continue
        if day_idx >= SMA_CMDTY:
            sma50 = np.mean(dbc_prices[day_idx - SMA_CMDTY:day_idx])
            deviation = (dbc_prices[day_idx] / sma50) - 1
            if deviation > 0:
                weight = min(deviation / 0.05, 1.0)
                r[i] = ret["DBC"][i] * weight
            else:
                r[i] = 0.0
        else:
            r[i] = ret["DBC"][i] * 0.5
    return r

def longvol_dual_mom(i_start=0):
    """BTAL with absolute momentum: if BTAL mom < 0, go to SHY."""
    r = np.zeros(N_ret)
    btal_ret = ret["BTAL"]
    shy_ret = ret["SHY"]
    use_btal = True
    for i in range(N_ret):
        d = dates_ret[i]
        is_rebal = (i == 0) or (d.month != dates_ret[i - 1].month)
        if is_rebal and not np.isnan(mom["BTAL"][i]):
            use_btal = mom["BTAL"][i] > ABS_MOM_THRESH
        if use_btal:
            r[i] = btal_ret[i] if not np.isnan(btal_ret[i]) else 0.0
        else:
            r[i] = shy_ret[i] if not np.isnan(shy_ret[i]) else 0.0
    return r


# --- Build component returns for each strategy ---
print("  Building Strategy Base...")
sel_base = run_base_selection()
comp_base = {
    "Equity":     block_returns(sel_base, "Equity"),
    "Bonds":      block_returns(sel_base, "Bonds"),
    "HardAssets": block_returns(sel_base, "HardAssets"),
    "LongVol":    longvol_returns(),
    "CmdtyTrend": cmdty_trend_returns(),
}

print("  Building Strategy A: Dual Momentum...")
sel_dual = run_dual_mom_selection()
comp_dualmom = {
    "Equity":     block_returns(sel_dual, "Equity"),
    "Bonds":      block_returns(sel_dual, "Bonds"),
    "HardAssets": block_returns(sel_dual, "HardAssets"),
    "LongVol":    longvol_dual_mom(),
    "CmdtyTrend": cmdty_trend_returns(),
}

print("  Building Strategy B: SMA200 Trend Filter...")
sel_sma, exp_sma = run_sma200_selection()
comp_sma200 = {
    "Equity":     block_returns(sel_sma, "Equity", exp_sma),
    "Bonds":      block_returns(sel_sma, "Bonds"),  # Bonds don't need SMA filter
    "HardAssets": block_returns(sel_sma, "HardAssets", exp_sma),
    "LongVol":    longvol_returns(),
    "CmdtyTrend": cmdty_trend_sma200(),
}

print("  Building Strategy C: Risk Parity...")
# Risk parity: inverse-vol weights within blocks, recalculated monthly
def risk_parity_block_returns(selections, block_name):
    """Inverse-volatility weighted returns for selected tickers."""
    r = np.zeros(N_ret)
    current_weights = None
    for i in range(N_ret):
        d = dates_ret[i]
        is_rebal = (i == 0) or (d.month != dates_ret[i - 1].month)
        if is_rebal:
            picks = selections[block_name][i]
            vols = []
            for t in picks:
                rv = rolling_vol[t][i] if t in rolling_vol and not np.isnan(rolling_vol[t][i]) else None
                vols.append(rv)
            # If we have valid vols, use inverse-vol weighting
            if all(v is not None and v > 0.001 for v in vols):
                inv_vols = [1.0 / v for v in vols]
                total_iv = sum(inv_vols)
                current_weights = {t: iv / total_iv for t, iv in zip(picks, inv_vols)}
            else:
                # Fallback to equal weight
                current_weights = {t: 1.0 / len(picks) for t in picks}

        if current_weights:
            day_ret = 0.0
            for t, w in current_weights.items():
                tr = ret[t][i] if t in ret and not np.isnan(ret[t][i]) else 0.0
                day_ret += w * tr
            r[i] = day_ret
    return r

comp_riskparity = {
    "Equity":     risk_parity_block_returns(sel_base, "Equity"),
    "Bonds":      risk_parity_block_returns(sel_base, "Bonds"),
    "HardAssets": risk_parity_block_returns(sel_base, "HardAssets"),
    "LongVol":    longvol_returns(),
    "CmdtyTrend": cmdty_trend_returns(),
}

# Risk parity also adjusts BLOCK weights by inverse-vol
# Compute rolling block volatilities and adjust
def risk_parity_portfolio(comp_returns, base_weights, dates_list):
    """Portfolio with risk-parity adjusted block weights."""
    comps = list(base_weights.keys())
    n = len(dates_list)
    port_ret = np.zeros(n)
    current_w = np.array([base_weights[c] for c in comps])

    for i in range(n):
        d = dates_list[i]
        is_rebal = (i == 0) or (d.month != dates_list[i - 1].month)

        if is_rebal and i >= VOL_LOOKBACK:
            # Compute rolling vol for each component
            block_vols = []
            for c in comps:
                window = comp_returns[c][max(0, i - VOL_LOOKBACK):i]
                valid = window[~np.isnan(window)]
                if len(valid) >= 20:
                    block_vols.append(np.std(valid) * np.sqrt(252))
                else:
                    block_vols.append(None)

            if all(v is not None and v > 0.001 for v in block_vols):
                inv_vols = [1.0 / v for v in block_vols]
                total_iv = sum(inv_vols)
                current_w = np.array([iv / total_iv for iv in inv_vols])
            else:
                current_w = np.array([base_weights[c] for c in comps])

        day_ret = sum(current_w[j] * comp_returns[comps[j]][i] for j in range(len(comps)))
        port_ret[i] = day_ret

    return port_ret

print("  Building Strategy D: Dual Momentum + Trend...")
sel_dual_trend = run_dual_mom_selection()
sel_sma_dt, exp_sma_dt = run_sma200_selection()
comp_dualtrend = {
    "Equity":     block_returns(sel_dual_trend, "Equity", exp_sma_dt),
    "Bonds":      block_returns(sel_dual_trend, "Bonds"),
    "HardAssets": block_returns(sel_dual_trend, "HardAssets", exp_sma_dt),
    "LongVol":    longvol_dual_mom(),
    "CmdtyTrend": cmdty_trend_sma200(),
}

# ═══════════════════════════════════════════════════════════════════
# 6. PORTFOLIO CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════
print("\nConstructing portfolios...")

def monthly_rebal_portfolio(weights_dict, comp_returns, dates_list):
    comps = list(weights_dict.keys())
    n = len(dates_list)
    target_w = np.array([weights_dict[c] for c in comps])
    alloc = target_w.copy()
    port_ret = np.zeros(n)
    for i in range(n):
        if i == 0 or dates_list[i].month != dates_list[i - 1].month:
            total_val = alloc.sum()
            alloc = target_w * total_val
        total_before = alloc.sum()
        for j, c in enumerate(comps):
            alloc[j] *= (1 + comp_returns[c][i])
        total_after = alloc.sum()
        port_ret[i] = (total_after / total_before) - 1 if total_before > 0 else 0
    return port_ret

# Build all strategies
strat_ret = {}

# Base
strat_ret["Base"] = monthly_rebal_portfolio(W_DRAGON, comp_base, dates_ret)

# A) Dual Momentum
strat_ret["DualMom"] = monthly_rebal_portfolio(W_DRAGON, comp_dualmom, dates_ret)

# B) SMA200 Trend
strat_ret["SMA200"] = monthly_rebal_portfolio(W_DRAGON, comp_sma200, dates_ret)

# C) Risk Parity (both within-block and across-block)
strat_ret["RiskParity"] = risk_parity_portfolio(comp_riskparity, W_DRAGON, dates_ret)

# D) Dual Mom + Trend (best individual filters combined)
strat_ret["DualTrend"] = monthly_rebal_portfolio(W_DRAGON, comp_dualtrend, dates_ret)

# 60/40 and SPY benchmarks
comp_6040 = {"Equity": ret["SPY"], "Bonds": ret["TLT"]}
strat_ret["6040"] = monthly_rebal_portfolio({"Equity": 0.60, "Bonds": 0.40}, comp_6040, dates_ret)
strat_ret["SPY"] = ret["SPY"]

# ═══════════════════════════════════════════════════════════════════
# 7. FIND BEST UNLEVERAGED STRATEGY & APPLY LEVERAGE
# ═══════════════════════════════════════════════════════════════════
print("\nFinding best strategy for leverage...")

def calc_metrics(returns, name=""):
    n = len(returns)
    years = n / 252
    total = np.prod(1 + returns) - 1
    cagr = (1 + total) ** (1 / years) - 1
    vol = np.std(returns) * np.sqrt(252)
    sharpe = (cagr - RF_ANNUAL) / vol if vol > 1e-8 else 0
    downside = returns[returns < 0]
    downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 1e-8
    sortino = (cagr - RF_ANNUAL) / downside_vol
    nav = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    mdd = np.min(dd) * 100
    calmar = cagr / abs(mdd / 100) if abs(mdd) > 0.01 else 0
    ret_to_risk = cagr / vol if vol > 1e-8 else 0

    # Rolling 1Y returns for win rate
    rolling_1y = []
    for i in range(252, n):
        r1y = np.prod(1 + returns[i - 252:i]) - 1
        rolling_1y.append(r1y)
    win_rate_1y = sum(1 for r in rolling_1y if r > 0) / len(rolling_1y) * 100 if rolling_1y else 0

    return {
        "name": name, "cagr": cagr, "vol": vol, "sharpe": sharpe,
        "sortino": sortino, "mdd": mdd, "calmar": calmar,
        "ret_to_risk": ret_to_risk, "total": total * 100, "years": years,
        "win_rate_1y": win_rate_1y,
    }

# Evaluate unlevered strategies
unlevered_keys = ["Base", "DualMom", "SMA200", "RiskParity", "DualTrend"]
metrics = {}
for key in unlevered_keys + ["6040", "SPY"]:
    metrics[key] = calc_metrics(strat_ret[key], STRAT_NAMES.get(key, key))

# Find best Sharpe
best_key = max(unlevered_keys, key=lambda k: metrics[k]["sharpe"])
print(f"  Best unlevered strategy: {STRAT_NAMES[best_key]} (Sharpe: {metrics[best_key]['sharpe']:.3f})")

# Apply leverage to best strategy
def apply_leverage(returns, leverage, daily_cost):
    """Apply leverage: levered_ret = leverage * asset_ret - (leverage - 1) * daily_cost"""
    levered = leverage * returns - (leverage - 1) * daily_cost
    return levered

strat_ret["Levered"] = apply_leverage(strat_ret[best_key], MAX_LEVERAGE, DEBT_COST_DAILY)
metrics["Levered"] = calc_metrics(strat_ret["Levered"], STRAT_NAMES["Levered"])

# Also test multiple leverage levels
leverage_sweep = {}
for lev in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
    lev_ret = apply_leverage(strat_ret[best_key], lev, DEBT_COST_DAILY)
    lev_m = calc_metrics(lev_ret, f"{lev:.1f}x")
    leverage_sweep[lev] = lev_m

print(f"\n  Leverage sweep on {STRAT_NAMES[best_key]}:")
for lev in sorted(leverage_sweep.keys()):
    m = leverage_sweep[lev]
    print(f"    {lev:.1f}x: CAGR {m['cagr']*100:+.1f}%  Vol {m['vol']*100:.1f}%  Sharpe {m['sharpe']:.3f}  MDD {m['mdd']:.1f}%")

# Find optimal leverage (maximize Sharpe)
opt_lev = max(leverage_sweep.keys(), key=lambda l: leverage_sweep[l]["sharpe"])
print(f"\n  Optimal leverage: {opt_lev:.1f}x (Sharpe: {leverage_sweep[opt_lev]['sharpe']:.3f})")

# Use optimal leverage if different from 1.5x
if opt_lev != MAX_LEVERAGE:
    strat_ret["Levered"] = apply_leverage(strat_ret[best_key], opt_lev, DEBT_COST_DAILY)
    metrics["Levered"] = calc_metrics(strat_ret["Levered"], f"E) Opt Lev {opt_lev:.1f}x")
    STRAT_NAMES["Levered"] = f"E) Opt Lev {opt_lev:.1f}x"

# Print summary
all_keys = unlevered_keys + ["Levered", "6040", "SPY"]
print(f"\n{'='*80}")
print(f"  {'Strategy':30s} {'CAGR':>8s} {'Vol':>8s} {'Sharpe':>8s} {'MDD':>8s} {'Sortino':>8s} {'Calmar':>8s}")
print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for key in all_keys:
    m = metrics[key]
    marker = " ***" if key == "Levered" else " <-- base" if key == "Base" else ""
    print(f"  {STRAT_NAMES.get(key, key):30s} {m['cagr']*100:+7.1f}% {m['vol']*100:7.1f}% {m['sharpe']:7.3f} {m['mdd']:7.1f}% {m['sortino']:7.2f} {m['calmar']:7.2f}{marker}")
print(f"{'='*80}")

# ═══════════════════════════════════════════════════════════════════
# 8. CUMULATIVE NAV
# ═══════════════════════════════════════════════════════════════════
def cum_nav(returns):
    nav = np.ones(len(returns) + 1)
    for i, r in enumerate(returns):
        nav[i + 1] = nav[i] * (1 + r)
    return nav

navs = {key: cum_nav(strat_ret[key]) for key in all_keys}

# ═══════════════════════════════════════════════════════════════════
# 9. ANNUAL RETURNS
# ═══════════════════════════════════════════════════════════════════
annual_returns = {}
for i, d in enumerate(dates_ret):
    yr = d.year
    if yr not in annual_returns:
        annual_returns[yr] = {k: [] for k in all_keys}
    for k in all_keys:
        annual_returns[yr][k].append(strat_ret[k][i])

annual_table = {}
for yr in sorted(annual_returns.keys()):
    annual_table[yr] = {}
    for k in all_keys:
        annual_table[yr][k] = (np.prod(1 + np.array(annual_returns[yr][k])) - 1) * 100

# ═══════════════════════════════════════════════════════════════════
# 10. STRESS PERIODS
# ═══════════════════════════════════════════════════════════════════
from datetime import datetime as _dt

STRESS_PERIODS = [
    ("Taper Tantrum",       "2013-05-22", "2013-06-24"),
    ("China / Oil Crash",   "2015-08-10", "2016-02-11"),
    ("Volmageddon",         "2018-01-26", "2018-02-08"),
    ("Q4 2018 Selloff",     "2018-10-01", "2018-12-24"),
    ("COVID Crash",         "2020-02-19", "2020-03-23"),
    ("2022 Bear / Rates",   "2022-01-03", "2022-10-12"),
    ("Trump Tariffs",       "2025-02-19", "2025-04-08"),
]

def period_return(returns, dates_list, start_str, end_str):
    start_d = _dt.strptime(start_str, "%Y-%m-%d").date()
    end_d = _dt.strptime(end_str, "%Y-%m-%d").date()
    mask = np.array([(d >= start_d and d <= end_d) for d in dates_list])
    if mask.sum() == 0:
        return None
    return (np.prod(1 + returns[mask]) - 1) * 100

stress_results = []
for sp_name, sp_start, sp_end in STRESS_PERIODS:
    r_base = period_return(strat_ret["Base"], dates_ret, sp_start, sp_end)
    if r_base is None:
        continue
    entry = {"name": sp_name, "start": sp_start, "end": sp_end}
    for k in all_keys:
        entry[k] = period_return(strat_ret[k], dates_ret, sp_start, sp_end)
    stress_results.append(entry)

# ═══════════════════════════════════════════════════════════════════
# 11. REGIME ANALYSIS
# ═══════════════════════════════════════════════════════════════════
REGIME_LOOKBACK = 252
regime_labels = []
for i in range(N_ret):
    day_idx = i + 1
    if day_idx >= REGIME_LOOKBACK:
        r1y = price_data["SPY"][day_idx] / price_data["SPY"][day_idx - REGIME_LOOKBACK] - 1
        if r1y > 0.15: regime_labels.append("Bull")
        elif r1y < -0.15: regime_labels.append("Bear")
        else: regime_labels.append("Flat")
    else:
        regime_labels.append("N/A")

regime_stats = {}
for regime in ["Bull", "Bear", "Flat"]:
    mask = np.array([r == regime for r in regime_labels])
    if mask.sum() < 20:
        continue
    regime_stats[regime] = {"days": int(mask.sum())}
    for k in all_keys:
        regime_stats[regime][k] = calc_metrics(strat_ret[k][mask], f"{STRAT_NAMES.get(k,k)} ({regime})")

# ═══════════════════════════════════════════════════════════════════
# 12. ROLLING SHARPE (3Y)
# ═══════════════════════════════════════════════════════════════════
ROLLING_WINDOW = 756  # 3 years
rolling_sharpe = {}
for k in ["Base", best_key, "Levered", "6040"]:
    rs = np.full(N_ret, np.nan)
    for i in range(ROLLING_WINDOW, N_ret):
        window = strat_ret[k][i - ROLLING_WINDOW:i]
        cagr_w = (np.prod(1 + window) ** (252 / len(window))) - 1
        vol_w = np.std(window) * np.sqrt(252)
        if vol_w > 0.001:
            rs[i] = (cagr_w - RF_ANNUAL) / vol_w
    rolling_sharpe[k] = rs

# ═══════════════════════════════════════════════════════════════════
# 13. SVG CHARTS
# ═══════════════════════════════════════════════════════════════════
print("\nGenerating charts...")

def svg_line(nav_arr, dates_arr, color, width=2.0, dashed=False, vw=720, vh=300,
             log_scale=True, y_min=None, y_max=None):
    n = len(nav_arr)
    if n == 0: return ""
    vals = np.log(np.clip(nav_arr, 1e-6, None)) if log_scale else np.array(nav_arr, dtype=float)
    if y_min is None: y_min = np.nanmin(vals)
    if y_max is None: y_max = np.nanmax(vals)
    y_range = y_max - y_min if y_max > y_min else 1
    ml, mr, mt, mb = 50, 15, 15, 30
    pw, ph = vw - ml - mr, vh - mt - mb
    pts = []
    for i in range(n):
        x = ml + (i / max(n - 1, 1)) * pw
        y = mt + ph - ((vals[i] - y_min) / y_range) * ph
        pts.append(f"{x:.1f},{y:.1f}")
    dash = ' stroke-dasharray="6,4"' if dashed else ""
    return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="{width}"{dash} stroke-linejoin="round" stroke-linecap="round"/>'

def svg_grid_and_labels(nav_dict, dates_arr, vw=720, vh=300, log_scale=True, n_y_labels=5):
    all_vals = []
    for nav in nav_dict.values():
        vals = np.log(np.clip(nav, 1e-6, None)) if log_scale else np.array(nav, dtype=float)
        all_vals.extend(vals[~np.isnan(vals)])
    y_min, y_max = min(all_vals), max(all_vals)
    y_range = y_max - y_min if y_max > y_min else 1
    ml, mr, mt, mb = 50, 15, 15, 30
    pw, ph = vw - ml - mr, vh - mt - mb
    svg = ""
    for i in range(n_y_labels + 1):
        frac = i / n_y_labels
        y_val = y_min + frac * y_range
        y_px = mt + ph - frac * ph
        dv = math.exp(y_val) if log_scale else y_val
        label = f"${dv:.1f}" if dv < 10 else f"${dv:.0f}"
        svg += f'<line x1="{ml}" y1="{y_px:.0f}" x2="{vw-mr}" y2="{y_px:.0f}" stroke="rgba(148,163,184,0.12)" stroke-width="0.5"/>'
        svg += f'<text x="{ml-5}" y="{y_px:.0f}" text-anchor="end" fill="#64748b" font-size="9" dominant-baseline="middle">{label}</text>'
    dl = dates_arr if isinstance(dates_arr, list) else list(dates_arr)
    nd = len(dl)
    seen = set()
    for i, d in enumerate(dl):
        yr = d.year if hasattr(d, "year") else d
        if yr not in seen and (i == 0 or yr != dl[i-1].year):
            seen.add(yr)
            x = ml + (i / max(nd - 1, 1)) * pw
            svg += f'<text x="{x:.0f}" y="{vh-5}" text-anchor="middle" fill="#64748b" font-size="8">{yr}</text>'
            svg += f'<line x1="{x:.0f}" y1="{mt}" x2="{x:.0f}" y2="{vh-mb}" stroke="rgba(148,163,184,0.06)" stroke-width="0.5"/>'
    return svg, y_min, y_max

def build_main_chart():
    vw, vh = 760, 340
    show = {k: navs[k] for k in ["Base", "DualMom", "SMA200", "RiskParity", "DualTrend", "Levered", "6040"]}
    g, ymn, ymx = svg_grid_and_labels(show, dates, vw, vh)
    lines = ""
    # Draw benchmarks first (behind), then strategies
    lines += svg_line(navs["6040"], dates, COLORS["6040"], 1.2, True, vw, vh, True, ymn, ymx)
    lines += svg_line(navs["Base"], dates, COLORS["Base"], 1.8, False, vw, vh, True, ymn, ymx)
    lines += svg_line(navs["DualMom"], dates, COLORS["DualMom"], 1.5, False, vw, vh, True, ymn, ymx)
    lines += svg_line(navs["SMA200"], dates, COLORS["SMA200"], 1.5, False, vw, vh, True, ymn, ymx)
    lines += svg_line(navs["RiskParity"], dates, COLORS["RiskParity"], 1.5, False, vw, vh, True, ymn, ymx)
    lines += svg_line(navs["DualTrend"], dates, COLORS["DualTrend"], 2.0, False, vw, vh, True, ymn, ymx)
    lines += svg_line(navs["Levered"], dates, COLORS["Levered"], 2.5, False, vw, vh, True, ymn, ymx)
    return f'<svg viewBox="0 0 {vw} {vh}" xmlns="http://www.w3.org/2000/svg">{g}{lines}</svg>'

def build_drawdown_chart():
    vw, vh = 760, 220
    def dd_s(r):
        n = np.cumprod(1 + r)
        p = np.maximum.accumulate(n)
        return ((n - p) / p) * 100
    dd_base = np.insert(dd_s(strat_ret["Base"]), 0, 0)
    dd_best = np.insert(dd_s(strat_ret[best_key]), 0, 0)
    dd_lev = np.insert(dd_s(strat_ret["Levered"]), 0, 0)
    dd_6040 = np.insert(dd_s(strat_ret["6040"]), 0, 0)
    ymn = min(np.min(dd_base), np.min(dd_best), np.min(dd_lev), np.min(dd_6040))
    ymx = 0
    ml, mr, mt, mb = 50, 15, 10, 25
    pw, ph = vw - ml - mr, vh - mt - mb
    yr = ymx - ymn if ymx > ymn else 1
    svg = ""
    for pct in [0, -10, -20, -30, -40, -50, -60]:
        if pct < ymn - 5:
            continue
        yp = mt + ph - ((pct - ymn) / yr) * ph
        svg += f'<line x1="{ml}" y1="{yp:.0f}" x2="{vw-mr}" y2="{yp:.0f}" stroke="rgba(148,163,184,0.12)" stroke-width="0.5"/>'
        svg += f'<text x="{ml-5}" y="{yp:.0f}" text-anchor="end" fill="#64748b" font-size="8" dominant-baseline="middle">{pct}%</text>'

    def dp(dd, color, w):
        pts = [f"{ml + (i / max(len(dd) - 1, 1)) * pw:.1f},{mt + ph - ((dd[i] - ymn) / yr) * ph:.1f}" for i in range(len(dd))]
        return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="{w}" stroke-linejoin="round"/>'
    svg += dp(dd_6040, COLORS["6040"], 1.0)
    svg += dp(dd_base, COLORS["Base"], 1.5)
    svg += dp(dd_best, COLORS[best_key], 1.8)
    svg += dp(dd_lev, COLORS["Levered"], 2.0)
    return f'<svg viewBox="0 0 {vw} {vh}" xmlns="http://www.w3.org/2000/svg">{svg}</svg>'

def build_rolling_sharpe_chart():
    vw, vh = 760, 220
    ml, mr, mt, mb = 50, 15, 10, 30
    pw, ph = vw - ml - mr, vh - mt - mb
    # Find y range
    all_vals = []
    for k in rolling_sharpe:
        valid = rolling_sharpe[k][~np.isnan(rolling_sharpe[k])]
        all_vals.extend(valid)
    if not all_vals:
        return '<svg viewBox="0 0 760 220"></svg>'
    ymn = max(min(all_vals), -2.0)
    ymx = min(max(all_vals), 3.0)
    yr = ymx - ymn if ymx > ymn else 1

    svg = ""
    # Grid
    for val in np.arange(math.floor(ymn), math.ceil(ymx) + 0.5, 0.5):
        if val < ymn or val > ymx:
            continue
        yp = mt + ph - ((val - ymn) / yr) * ph
        c = "rgba(148,163,184,0.25)" if val == 0 else "rgba(148,163,184,0.08)"
        svg += f'<line x1="{ml}" y1="{yp:.0f}" x2="{vw-mr}" y2="{yp:.0f}" stroke="{c}" stroke-width="0.5"/>'
        svg += f'<text x="{ml-5}" y="{yp:.0f}" text-anchor="end" fill="#64748b" font-size="8" dominant-baseline="middle">{val:.1f}</text>'

    # Lines
    for k in rolling_sharpe:
        rs = rolling_sharpe[k]
        color = COLORS.get(k, "#94a3b8")
        w = 2.0 if k == "Levered" else 1.5 if k == best_key else 1.2
        pts = []
        for i in range(N_ret):
            if np.isnan(rs[i]):
                continue
            x = ml + (i / max(N_ret - 1, 1)) * pw
            y = mt + ph - ((np.clip(rs[i], ymn, ymx) - ymn) / yr) * ph
            pts.append(f"{x:.1f},{y:.1f}")
        if pts:
            dashed = ' stroke-dasharray="4,3"' if k == "6040" else ""
            svg += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="{w}"{dashed} stroke-linejoin="round"/>'

    # Year labels
    seen = set()
    for i, d in enumerate(dates_ret):
        if d.year not in seen and (i == 0 or d.year != dates_ret[i-1].year):
            seen.add(d.year)
            x = ml + (i / max(N_ret - 1, 1)) * pw
            svg += f'<text x="{x:.0f}" y="{vh-5}" text-anchor="middle" fill="#64748b" font-size="8">{d.year}</text>'

    return f'<svg viewBox="0 0 {vw} {vh}" xmlns="http://www.w3.org/2000/svg">{svg}</svg>'

def build_leverage_chart():
    """Bar chart showing Sharpe at different leverage levels."""
    vw, vh = 400, 200
    ml, mr, mt, mb = 50, 20, 15, 35
    pw, ph = vw - ml - mr, vh - mt - mb
    levels = sorted(leverage_sweep.keys())
    n = len(levels)
    bar_w = pw / n * 0.7
    gap = pw / n
    max_sharpe = max(m["sharpe"] for m in leverage_sweep.values())
    min_sharpe = min(0, min(m["sharpe"] for m in leverage_sweep.values()))
    sr = max_sharpe - min_sharpe if max_sharpe > min_sharpe else 1

    svg = ""
    # Zero line
    zero_y = mt + ph - ((0 - min_sharpe) / sr) * ph
    svg += f'<line x1="{ml}" y1="{zero_y:.0f}" x2="{vw-mr}" y2="{zero_y:.0f}" stroke="rgba(148,163,184,0.2)" stroke-width="1"/>'

    for idx, lev in enumerate(levels):
        m = leverage_sweep[lev]
        x = ml + idx * gap + (gap - bar_w) / 2
        h = ((m["sharpe"] - min_sharpe) / sr) * ph
        y = mt + ph - h
        is_opt = lev == opt_lev
        color = "#ef4444" if is_opt else "rgba(148,163,184,0.3)"
        svg += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" rx="3"/>'
        svg += f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" fill="{"#ef4444" if is_opt else "#94a3b8"}" font-size="9" font-weight="{"700" if is_opt else "400"}">{m["sharpe"]:.2f}</text>'
        svg += f'<text x="{x + bar_w / 2:.1f}" y="{vh - 10:.1f}" text-anchor="middle" fill="#64748b" font-size="9">{lev:.1f}x</text>'

    return f'<svg viewBox="0 0 {vw} {vh}" xmlns="http://www.w3.org/2000/svg">{svg}</svg>'

# ═══════════════════════════════════════════════════════════════════
# 14. HTML REPORT
# ═══════════════════════════════════════════════════════════════════
print("\nGenerating HTML report...")

def pct_cls(v):
    if isinstance(v, float) and np.isnan(v):
        return ""
    return "pos" if v > 0 else "neg" if v < 0 else ""

# Sharpe improvement
base_sharpe = metrics["Base"]["sharpe"]
best_unlevered = max(unlevered_keys, key=lambda k: metrics[k]["sharpe"])
best_sharpe = metrics[best_unlevered]["sharpe"]
levered_sharpe = metrics["Levered"]["sharpe"]
improvement_unlevered = (best_sharpe / base_sharpe - 1) * 100 if base_sharpe > 0 else 0
improvement_levered = (levered_sharpe / base_sharpe - 1) * 100 if base_sharpe > 0 else 0

def strategy_desc(key):
    descs = {
        "DualMom": "Filtra activos con momentum absoluto negativo → reemplaza con SHY (cash proxy). Reduce exposicion en tendencias bajistas.",
        "SMA200": "Si un activo esta debajo de su SMA200, reduce su peso proporcional (min 30%). Filtra regimenes bajistas a nivel de activo.",
        "RiskParity": "Pesos por inversa de volatilidad rolling 60d, tanto intra-bloque como inter-bloque. Equaliza contribucion de riesgo.",
        "DualTrend": "Combina Dual Momentum (filtro absoluto) + SMA200 Trend Filter. Doble proteccion contra drawdowns.",
        "Levered": f"Aplica {opt_lev:.1f}x leverage sobre {STRAT_NAMES[best_key]} con costo de deuda {DEBT_COST_ANNUAL*100:.0f}% anual. Amplifica retorno y riesgo.",
    }
    return descs.get(key, "")

def strat_kpi_card(key):
    m = metrics[key]
    color = COLORS[key]
    delta_sharpe = (m["sharpe"] / base_sharpe - 1) * 100 if base_sharpe > 0 else 0
    delta_cls = "pos" if delta_sharpe > 0 else "neg"
    return f'''
    <div class="strat-card" style="border-left:3px solid {color}">
      <div class="strat-header">
        <span style="color:{color};font-weight:800;font-size:13px">{STRAT_NAMES[key]}</span>
        <span class="strat-delta {delta_cls}">{delta_sharpe:+.0f}% Sharpe</span>
      </div>
      <div class="strat-desc">{strategy_desc(key)}</div>
      <div class="strat-kpis">
        <div><span class="sk-label">CAGR</span><span class="sk-value {pct_cls(m['cagr'])}">{m['cagr']*100:+.1f}%</span></div>
        <div><span class="sk-label">Vol</span><span class="sk-value">{m['vol']*100:.1f}%</span></div>
        <div><span class="sk-label">Sharpe</span><span class="sk-value" style="color:{color}">{m['sharpe']:.3f}</span></div>
        <div><span class="sk-label">Sortino</span><span class="sk-value">{m['sortino']:.2f}</span></div>
        <div><span class="sk-label">MDD</span><span class="sk-value neg">{m['mdd']:.1f}%</span></div>
        <div><span class="sk-label">Calmar</span><span class="sk-value">{m['calmar']:.2f}</span></div>
      </div>
    </div>'''

def regime_rows():
    rows = ""
    for regime in ["Bull", "Bear", "Flat"]:
        if regime not in regime_stats:
            continue
        rs = regime_stats[regime]
        sym = "+" if regime == "Bull" else "-" if regime == "Bear" else "="
        clr = "#10b981" if regime == "Bull" else "#ef4444" if regime == "Bear" else "#94a3b8"
        rows += f'<tr><td style="color:{clr};font-weight:700">{sym} {regime}</td><td class="num">{rs["days"]}</td>'
        for k in ["Base", best_key, "Levered", "6040"]:
            m = rs[k]
            rows += f'<td class="num {pct_cls(m["cagr"])}" style="{"font-weight:700" if k=="Levered" else ""}">{m["cagr"]*100:+.1f}%</td>'
        rows += '</tr>'
    return rows

def annual_rows():
    rows = ""
    for yr in sorted(annual_table.keys()):
        a = annual_table[yr]
        rows += f'<tr><td>{yr}</td>'
        for k in ["Base", "DualMom", "SMA200", "DualTrend", "Levered", "6040", "SPY"]:
            fw = "font-weight:600;" if k == "Levered" else ""
            rows += f'<td class="num {pct_cls(a[k]/100)}" style="{fw}">{a[k]:+.1f}%</td>'
        rows += '</tr>'
    return rows

# Build legend for main chart
def main_legend():
    items = []
    for k in ["Base", "DualMom", "SMA200", "RiskParity", "DualTrend", "Levered", "6040"]:
        name = STRAT_NAMES.get(k, k)
        c = COLORS[k]
        final = f"${navs[k][-1]:.2f}"
        fw = "font-weight:700;" if k in ["Levered", best_key] else ""
        items.append(f'<div class="legend-item"><div class="legend-dot" style="background:{c}"></div><span style="color:{c};{fw}">{name}</span> <span style="color:#64748b">{final}</span></div>')
    return "\n".join(items)

html = f'''<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dragon Portfolio — Alternativas | SFinance</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: 'Inter', -apple-system, sans-serif; background:#0f172a; color:#e2e8f0; }}
.container {{ max-width:1200px; margin:0 auto; padding:24px 20px; }}
.header {{ display:flex; justify-content:space-between; align-items:center; padding:16px 0; border-bottom:1px solid rgba(148,163,184,0.12); margin-bottom:24px; }}
.header-title {{ font-size:22px; font-weight:800; letter-spacing:-0.5px; }}
.header-title span {{ color:#ef4444; }}
.header-sub {{ font-size:11px; color:#64748b; text-align:right; }}
.header-sub strong {{ color:#94a3b8; }}

.hero {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-bottom:28px; }}
.hero-card {{ background:rgba(30,41,59,0.6); border:1px solid rgba(148,163,184,0.08); border-radius:8px; padding:20px; text-align:center; }}
.hero-label {{ font-size:9px; text-transform:uppercase; color:#64748b; letter-spacing:0.5px; margin-bottom:6px; }}
.hero-value {{ font-size:28px; font-weight:900; letter-spacing:-1px; }}
.hero-sub {{ font-size:10px; color:#475569; margin-top:4px; }}

.strat-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:24px; }}
.strat-card {{ background:rgba(30,41,59,0.4); border:1px solid rgba(148,163,184,0.08); border-radius:8px; padding:14px 16px; }}
.strat-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }}
.strat-delta {{ font-size:10px; font-weight:700; padding:2px 8px; border-radius:4px; }}
.strat-delta.pos {{ background:rgba(16,185,129,0.12); color:#10b981; }}
.strat-delta.neg {{ background:rgba(239,68,68,0.12); color:#ef4444; }}
.strat-desc {{ font-size:9px; color:#94a3b8; line-height:1.5; margin-bottom:10px; }}
.strat-kpis {{ display:grid; grid-template-columns:repeat(6, 1fr); gap:6px; }}
.strat-kpis div {{ text-align:center; }}
.sk-label {{ display:block; font-size:7px; text-transform:uppercase; color:#475569; }}
.sk-value {{ display:block; font-size:13px; font-weight:700; }}

.section-title {{ font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; color:#94a3b8; margin-bottom:12px; }}
.chart-container {{ background:rgba(15,23,42,0.5); border:1px solid rgba(148,163,184,0.06); border-radius:6px; padding:12px; margin-bottom:8px; }}
.legend-row {{ display:flex; gap:14px; justify-content:center; padding:8px 0; flex-wrap:wrap; }}
.legend-item {{ display:flex; align-items:center; gap:5px; font-size:9px; color:#94a3b8; }}
.legend-dot {{ width:8px; height:8px; border-radius:50%; flex-shrink:0; }}
.section {{ margin-bottom:24px; }}
.card {{ background:rgba(30,41,59,0.4); border:1px solid rgba(148,163,184,0.08); border-radius:8px; padding:16px; }}
.grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:24px; }}
.grid-3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-bottom:24px; }}

.data-table {{ width:100%; border-collapse:collapse; font-size:10px; }}
.data-table th {{ background:rgba(30,41,59,0.8); color:#94a3b8; font-weight:600; text-transform:uppercase; font-size:8px; letter-spacing:0.5px; padding:8px 6px; text-align:left; border-bottom:1px solid rgba(148,163,184,0.12); position:sticky; top:0; z-index:1; }}
.data-table td {{ padding:6px; border-bottom:1px solid rgba(148,163,184,0.06); }}
.data-table .num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.data-table tr:hover {{ background:rgba(148,163,184,0.04); }}
.table-scroll {{ overflow-x:auto; max-height:400px; overflow-y:auto; }}

.pos {{ color:#10b981; }} .neg {{ color:#ef4444; }}
.footer {{ display:flex; justify-content:space-between; font-size:8px; color:#475569; padding:16px 0; border-top:1px solid rgba(148,163,184,0.08); margin-top:20px; }}
@media (max-width: 900px) {{
  .hero {{ grid-template-columns:1fr; }}
  .strat-grid {{ grid-template-columns:1fr; }}
  .grid-2,.grid-3 {{ grid-template-columns:1fr; }}
  .strat-kpis {{ grid-template-columns:repeat(3, 1fr); }}
}}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <div>
      <div class="header-title">Dragon Portfolio — <span>Alternativas</span></div>
      <div style="font-size:10px;color:#64748b;margin-top:2px">Objetivo: Doblar Sharpe Ratio | Leverage hasta {MAX_LEVERAGE:.1f}x @ {DEBT_COST_ANNUAL*100:.0f}% | {len(ALL_TICKERS)} activos | {dates[0]} → {dates[-1]}</div>
    </div>
    <div class="header-sub"><strong>SFinance-alicIA</strong><br>{TODAY.strftime("%d %b %Y")}</div>
  </div>

  <!-- HERO: Sharpe comparison -->
  <div class="hero">
    <div class="hero-card">
      <div class="hero-label">Sharpe — Dragon Base</div>
      <div class="hero-value" style="color:#94a3b8">{base_sharpe:.3f}</div>
      <div class="hero-sub">CAGR {metrics['Base']['cagr']*100:+.1f}% | Vol {metrics['Base']['vol']*100:.1f}% | MDD {metrics['Base']['mdd']:.1f}%</div>
    </div>
    <div class="hero-card" style="border:1px solid rgba(16,185,129,0.2)">
      <div class="hero-label">Sharpe — Mejor sin Leverage</div>
      <div class="hero-value" style="color:#10b981">{best_sharpe:.3f}</div>
      <div class="hero-sub">{STRAT_NAMES[best_unlevered]} | <span class="pos">+{improvement_unlevered:.0f}%</span> vs Base</div>
    </div>
    <div class="hero-card" style="border:1px solid rgba(239,68,68,0.3);background:rgba(239,68,68,0.04)">
      <div class="hero-label">Sharpe — Optimized + Leverage</div>
      <div class="hero-value" style="color:#ef4444">{levered_sharpe:.3f}</div>
      <div class="hero-sub">{STRAT_NAMES['Levered']} | <span style="color:#ef4444;font-weight:700">+{improvement_levered:.0f}%</span> vs Base</div>
    </div>
  </div>

  <!-- STRATEGY CARDS -->
  <div class="section">
    <div class="section-title">Estrategias Evaluadas</div>
    <div class="strat-grid">
      {strat_kpi_card("DualMom")}
      {strat_kpi_card("SMA200")}
      {strat_kpi_card("RiskParity")}
      {strat_kpi_card("DualTrend")}
    </div>
    <div style="max-width:600px;margin:0 auto">
      {strat_kpi_card("Levered")}
    </div>
  </div>

  <!-- MAIN CHART -->
  <div class="section">
    <div class="section-title">Crecimiento de $1 — Todas las Estrategias (Log)</div>
    <div class="chart-container">{build_main_chart()}</div>
    <div class="legend-row">{main_legend()}</div>
  </div>

  <!-- DRAWDOWN -->
  <div class="section">
    <div class="section-title">Drawdown Comparativo</div>
    <div class="chart-container">{build_drawdown_chart()}</div>
    <div class="legend-row">
      <div class="legend-item"><div class="legend-dot" style="background:{COLORS['Base']}"></div><span style="color:{COLORS['Base']}">Base</span> {metrics['Base']['mdd']:.1f}%</div>
      <div class="legend-item"><div class="legend-dot" style="background:{COLORS[best_key]}"></div><span style="color:{COLORS[best_key]}">{STRAT_NAMES[best_key]}</span> {metrics[best_key]['mdd']:.1f}%</div>
      <div class="legend-item"><div class="legend-dot" style="background:{COLORS['Levered']}"></div><span style="color:{COLORS['Levered']}">Levered</span> {metrics['Levered']['mdd']:.1f}%</div>
      <div class="legend-item"><div class="legend-dot" style="background:{COLORS['6040']}"></div><span style="color:{COLORS['6040']}">60/40</span> {metrics['6040']['mdd']:.1f}%</div>
    </div>
  </div>

  <!-- ROLLING SHARPE -->
  <div class="section">
    <div class="section-title">Rolling Sharpe Ratio (3 Anos)</div>
    <div class="chart-container">{build_rolling_sharpe_chart()}</div>
    <div class="legend-row">
      <div class="legend-item"><div class="legend-dot" style="background:{COLORS['Base']}"></div><span style="color:{COLORS['Base']}">Base</span></div>
      <div class="legend-item"><div class="legend-dot" style="background:{COLORS[best_key]}"></div><span style="color:{COLORS[best_key]}">{STRAT_NAMES[best_key]}</span></div>
      <div class="legend-item"><div class="legend-dot" style="background:{COLORS['Levered']}"></div><span style="color:{COLORS['Levered']}">Levered</span></div>
      <div class="legend-item"><div class="legend-dot" style="background:{COLORS['6040']}"></div><span style="color:{COLORS['6040']}">60/40</span></div>
    </div>
  </div>

  <!-- LEVERAGE SWEEP -->
  <div class="grid-2">
    <div class="card">
      <div class="section-title">Leverage Sweep — {STRAT_NAMES[best_key]} @ {DEBT_COST_ANNUAL*100:.0f}% costo deuda</div>
      <div style="text-align:center">{build_leverage_chart()}</div>
      <div style="font-size:9px;color:#64748b;text-align:center;margin-top:6px">
        Optimo: <strong style="color:#ef4444">{opt_lev:.1f}x</strong> (Sharpe {leverage_sweep[opt_lev]['sharpe']:.3f}) |
        CAGR {leverage_sweep[opt_lev]['cagr']*100:+.1f}% | MDD {leverage_sweep[opt_lev]['mdd']:.1f}%
      </div>
    </div>
    <div class="card">
      <div class="section-title">Performance por Regimen (CAGR anualizado)</div>
      <table class="data-table">
        <tr><th>Regimen</th><th class="num">Dias</th>
        <th class="num" style="color:{COLORS['Base']}">Base</th>
        <th class="num" style="color:{COLORS[best_key]}">{STRAT_NAMES[best_key][:8]}</th>
        <th class="num" style="color:{COLORS['Levered']}">Levered</th>
        <th class="num" style="color:{COLORS['6040']}">60/40</th></tr>
        {regime_rows()}
      </table>
    </div>
  </div>

  <!-- STRESS PERIODS -->
  <div class="section">
    <div class="section-title">Periodos de Stress — Retorno Total</div>
    <table class="data-table">
      <tr>
        <th>Evento</th><th>Periodo</th>
        <th class="num" style="color:{COLORS['Base']}">Base</th>
        <th class="num" style="color:{COLORS['DualMom']}">Dual Mom</th>
        <th class="num" style="color:{COLORS['SMA200']}">SMA200</th>
        <th class="num" style="color:{COLORS['DualTrend']}">Dual+Trend</th>
        <th class="num" style="color:{COLORS['Levered']}">Levered</th>
        <th class="num" style="color:{COLORS['6040']}">60/40</th>
        <th class="num" style="color:{COLORS['SPY']}">S&P 500</th>
      </tr>
      {''.join(
          f'<tr>'
          f'<td style="font-weight:700;white-space:nowrap">{sr["name"]}</td>'
          f'<td style="color:#64748b;font-size:8px;white-space:nowrap">{sr["start"]} → {sr["end"]}</td>'
          + ''.join(
              f'<td class="num" style="{"font-weight:700;" if k=="Levered" else ""}color:{"#10b981" if (sr[k] or 0)>=0 else "#ef4444"}">{sr[k]:+.1f}%</td>'
              if sr[k] is not None else '<td class="num">--</td>'
              for k in ["Base", "DualMom", "SMA200", "DualTrend", "Levered", "6040", "SPY"]
          )
          + f'</tr>'
          for sr in stress_results
      )}
    </table>
  </div>

  <!-- ANNUAL RETURNS -->
  <div class="section">
    <div class="section-title">Retornos Anuales</div>
    <div class="table-scroll">
    <table class="data-table">
      <tr><th>Ano</th>
        <th class="num" style="color:{COLORS['Base']}">Base</th>
        <th class="num" style="color:{COLORS['DualMom']}">Dual Mom</th>
        <th class="num" style="color:{COLORS['SMA200']}">SMA200</th>
        <th class="num" style="color:{COLORS['DualTrend']}">Dual+Trend</th>
        <th class="num" style="color:{COLORS['Levered']}">Levered</th>
        <th class="num" style="color:{COLORS['6040']}">60/40</th>
        <th class="num" style="color:{COLORS['SPY']}">S&P 500</th>
      </tr>
      {annual_rows()}
    </table>
    </div>
  </div>

  <!-- FULL STATS TABLE -->
  <div class="section">
    <div class="section-title">Estadisticas Completas</div>
    <div class="table-scroll">
    <table class="data-table">
      <tr><th>Metrica</th>
        {''.join(f'<th class="num" style="color:{COLORS.get(k, "#94a3b8")}">{STRAT_NAMES.get(k, k)}</th>' for k in all_keys)}
      </tr>
      <tr><td>CAGR</td>{''.join(f'<td class="num {pct_cls(metrics[k]["cagr"])}" style="{"font-weight:700" if k=="Levered" else ""}">{metrics[k]["cagr"]*100:+.1f}%</td>' for k in all_keys)}</tr>
      <tr><td>Volatilidad</td>{''.join(f'<td class="num">{metrics[k]["vol"]*100:.1f}%</td>' for k in all_keys)}</tr>
      <tr><td>Sharpe</td>{''.join(f'<td class="num" style="{"font-weight:700;color:" + COLORS.get(k,"#94a3b8") if k in ["Levered", best_key] else ""}">{metrics[k]["sharpe"]:.3f}</td>' for k in all_keys)}</tr>
      <tr><td>Sortino</td>{''.join(f'<td class="num">{metrics[k]["sortino"]:.2f}</td>' for k in all_keys)}</tr>
      <tr><td>Max Drawdown</td>{''.join(f'<td class="num neg">{metrics[k]["mdd"]:.1f}%</td>' for k in all_keys)}</tr>
      <tr><td>Calmar</td>{''.join(f'<td class="num">{metrics[k]["calmar"]:.2f}</td>' for k in all_keys)}</tr>
      <tr><td>Ret/Risk</td>{''.join(f'<td class="num">{metrics[k]["ret_to_risk"]:.2f}x</td>' for k in all_keys)}</tr>
      <tr><td>Win Rate 1Y</td>{''.join(f'<td class="num">{metrics[k]["win_rate_1y"]:.0f}%</td>' for k in all_keys)}</tr>
      <tr><td>Total Return</td>{''.join(f'<td class="num {pct_cls(metrics[k]["total"])}" style="{"font-weight:700" if k=="Levered" else ""}">{metrics[k]["total"]:+.0f}%</td>' for k in all_keys)}</tr>
      <tr><td>$1 Final</td>{''.join(f'<td class="num" style="{"font-weight:700" if k=="Levered" else ""}">${navs[k][-1]:.2f}</td>' for k in all_keys)}</tr>
    </table>
    </div>
  </div>

  <!-- METHODOLOGY -->
  <div class="card" style="margin-bottom:20px">
    <div class="section-title">Metodologia</div>
    <div style="font-size:9px;color:#94a3b8;line-height:1.7;columns:2;column-gap:24px">
      <p><strong style="color:#e2e8f0">Base</strong> — Dragon Portfolio v2 original: top-3 momentum {MOM_LOOKBACK}d, rebalanceo mensual, pesos fijos (Eq 24%, Bonds 18%, Hard 19%, LVol 21%, Cmdty 18%).</p>
      <p style="margin-top:6px"><strong style="color:{COLORS['DualMom']}">A) Dual Momentum</strong> — Si el mejor activo seleccionado tiene momentum absoluto &lt; 0, se reemplaza por SHY. BTAL tambien aplica filtro absoluto.</p>
      <p style="margin-top:6px"><strong style="color:{COLORS['SMA200']}">B) SMA200 Trend</strong> — Exposicion proporcional a cuantos picks estan sobre su SMA200 (min 30%). DBC requiere &gt; SMA200 para activar commodity trend.</p>
      <p style="margin-top:6px"><strong style="color:{COLORS['RiskParity']}">C) Risk Parity</strong> — Pesos intra-bloque e inter-bloque por inversa de volatilidad rolling {VOL_LOOKBACK}d. Recalibrado mensual.</p>
      <p style="margin-top:6px"><strong style="color:{COLORS['DualTrend']}">D) Dual Mom + Trend</strong> — Combina filtro de momentum absoluto con SMA200. Maxima proteccion contra drawdowns.</p>
      <p style="margin-top:6px"><strong style="color:{COLORS['Levered']}">E) Optimized Levered</strong> — {opt_lev:.1f}x leverage sobre {STRAT_NAMES[best_key]}. Costo deuda: {DEBT_COST_ANNUAL*100:.0f}% anual ({DEBT_COST_DAILY*10000:.2f} bps/dia).</p>
      <p style="margin-top:6px"><strong style="color:#e2e8f0">Rf</strong> — {RF_ANNUAL*100:.1f}% (T-bills). <strong style="color:#e2e8f0">Universo</strong> — {len(ALL_TICKERS)} activos, {N_ret} trading days.</p>
    </div>
  </div>

  <div class="footer">
    <span>SFinance-alicIA | Dragon Portfolio — Alternativas | Solo fines informativos, no es asesoria financiera</span>
    <span>{TODAY.strftime("%Y-%m-%d")} | Leverage max {MAX_LEVERAGE:.1f}x @ {DEBT_COST_ANNUAL*100:.0f}% costo deuda</span>
  </div>

</div>
</body>
</html>'''

# ═══════════════════════════════════════════════════════════════════
# 15. OUTPUT
# ═══════════════════════════════════════════════════════════════════
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dragon_Alternativas.html")
with open(outpath, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\n  Report saved: {outpath}")

if not os.environ.get("CI"):
    os.system(f'open "{outpath}"')
print("\n=== Done ===")
