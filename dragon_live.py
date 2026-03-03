#!/usr/bin/env python3
"""
Dragon Portfolio v3 — Live Dashboard
Centinela v3 SMA200 — Current state, YTD evolution, signals.
SFinance-alicIA
"""
import yfinance as yf
import numpy as np
import datetime, os, math, json

# ═══════════════════════════════════════════════════════════════════
# 1. CONFIG
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
SMA_LONG = 200
SMA_CMDTY = 50
MIN_EXPOSURE = 0.30

START = "2006-03-01"
TODAY = datetime.date.today()
RF_ANNUAL = 0.043
BTAL_LEVERAGE = 1.0

W_DRAGON = {"Equity": 0.24, "Bonds": 0.18, "HardAssets": 0.19, "LongVol": 0.21, "CmdtyTrend": 0.18}

TICKER_LABELS = {
    "SPY": "US Large", "QQQ": "Nasdaq", "IWM": "Small Cap",
    "EEM": "Emergentes", "VGK": "Europa", "EWY": "Korea",
    "EWP": "Espana", "EWZ": "Brasil", "EPOL": "Polonia",
    "SHY": "1-3Y", "IEF": "7-10Y", "TLT": "20+Y", "TIP": "TIPS", "LQD": "IG Corp",
    "GLD": "Oro", "SLV": "Plata", "CPER": "Cobre", "BTC-USD": "Bitcoin",
    "BTAL": "Anti-Beta", "DBC": "Commodities",
}
TICKER_COLORS = {
    "SPY": "#3b82f6", "QQQ": "#8b5cf6", "IWM": "#f97316",
    "EEM": "#f59e0b", "VGK": "#06b6d4", "EWY": "#ec4899",
    "EWP": "#ef4444", "EWZ": "#22c55e", "EPOL": "#a855f7",
    "SHY": "#94a3b8", "IEF": "#38bdf8", "TLT": "#0ea5e9", "TIP": "#f97316", "LQD": "#10b981",
    "GLD": "#f59e0b", "SLV": "#94a3b8", "CPER": "#f97316", "BTC-USD": "#f7931a",
    "BTAL": "#ef4444", "DBC": "#a855f7",
}
BLOCK_COLORS = {
    "Equity": "#10b981", "Bonds": "#06b6d4", "HardAssets": "#f59e0b",
    "LongVol": "#ef4444", "CmdtyTrend": "#a855f7",
}
BLOCK_LABELS = {
    "Equity": "Equity", "Bonds": "Bonds", "HardAssets": "Hard Assets",
    "LongVol": "Long Vol", "CmdtyTrend": "Cmdty Trend",
}

# ═══════════════════════════════════════════════════════════════════
# 2. DATA FETCHING
# ═══════════════════════════════════════════════════════════════════
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "price_cache.json")
print("═══ Centinela v3 — Live Dashboard ═══\n")

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
    for t in ALL_TICKERS:
        if t not in price_data:
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

print(f"  Aligned: {N} trading days ({dates[0]} -> {dates[-1]})")

# Daily returns
ret = {}
for t in ALL_TICKERS:
    p = price_data[t]
    ret[t] = np.diff(p) / p[:-1]
dates_ret = dates[1:]
N_ret = len(dates_ret)

# ═══════════════════════════════════════════════════════════════════
# 3. MOMENTUM + SMA200
# ═══════════════════════════════════════════════════════════════════
print("Computing momentum + SMA200...")

mom = {}
for t in ALL_TICKERS:
    p = price_data[t]
    m = np.full(N_ret, np.nan)
    for i in range(N_ret):
        if i >= MOM_LOOKBACK:
            p_now = p[i]
            p_prev = p[i - MOM_LOOKBACK]
            if not np.isnan(p_now) and not np.isnan(p_prev) and p_prev > 0:
                m[i] = p_now / p_prev - 1
    mom[t] = m

# SMA200 signals
sma200_above = {}
sma200_values = {}
for t in ALL_TICKERS:
    p = price_data[t]
    signal = np.full(N, False)
    sma_v = np.full(N, np.nan)
    for i in range(N):
        if i >= SMA_LONG:
            window = p[i - SMA_LONG + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= SMA_LONG * 0.8:
                sma_v[i] = np.mean(valid)
                signal[i] = p[i] > sma_v[i]
    sma200_above[t] = signal[1:]  # align with returns
    sma200_values[t] = sma_v

# ═══════════════════════════════════════════════════════════════════
# 4. SELECTION + SMA200 EXPOSURE SCALING
# ═══════════════════════════════════════════════════════════════════
selections = {block: [] for block in UNIVERSES}
exposure_scale = {block: [] for block in UNIVERSES}
current_sel = {}
for block, candidates in UNIVERSES.items():
    ns = min(N_SELECT, len(candidates))
    defaults = [t for t in candidates if t not in LATE_JOINERS][:ns]
    if len(defaults) < ns:
        defaults = candidates[:ns]
    current_sel[block] = defaults

selection_log = []
for i in range(N_ret):
    d = dates_ret[i]
    is_rebal = (i == 0) or (d.month != dates_ret[i - 1].month)
    if is_rebal:
        log_entry = {"date": d}
        for block, candidates in UNIVERSES.items():
            ns = min(N_SELECT, len(candidates))
            scores = {t: mom[t][i] for t in candidates}
            valid_candidates = [(t, scores[t]) for t in candidates if not np.isnan(scores[t])]
            valid_candidates.sort(key=lambda x: -x[1])
            if len(valid_candidates) >= ns:
                current_sel[block] = [t for t, _ in valid_candidates[:ns]]
            elif len(valid_candidates) > 0:
                current_sel[block] = [t for t, _ in valid_candidates]
            log_entry[block] = {"picks": list(current_sel[block]), "scores": scores}
        selection_log.append(log_entry)
    for block in UNIVERSES:
        selections[block].append(list(current_sel[block]))
        picks_b = current_sel[block]
        above_b = sum(1 for t in picks_b if t in sma200_above and sma200_above[t][i])
        sc = above_b / max(len(picks_b), 1)
        sc = max(sc, MIN_EXPOSURE)
        exposure_scale[block].append(sc)

# ═══════════════════════════════════════════════════════════════════
# 5. STRATEGY CONSTRUCTION (SMA200)
# ═══════════════════════════════════════════════════════════════════
print("Building portfolio (SMA200 filtered)...")

shy_ret = ret["SHY"]

def dynamic_block_returns_sma200(block_name, use_sma_filter=True):
    r = np.zeros(N_ret)
    for i in range(N_ret):
        picks = selections[block_name][i]
        valid = [ret[t][i] for t in picks if not np.isnan(ret[t][i])]
        risk_ret = np.mean(valid) if valid else 0.0
        if use_sma_filter:
            sc = exposure_scale[block_name][i]
            shy_r = shy_ret[i] if not np.isnan(shy_ret[i]) else 0.0
            r[i] = sc * risk_ret + (1 - sc) * shy_r
        else:
            r[i] = risk_ret
    return r

ret_equity_sma = dynamic_block_returns_sma200("Equity", True)
ret_bonds = dynamic_block_returns_sma200("Bonds", False)
ret_hard_sma = dynamic_block_returns_sma200("HardAssets", True)
ret_longvol = ret["BTAL"] * BTAL_LEVERAGE

dbc_prices = price_data["DBC"]
ret_cmdty_trend = np.zeros(N_ret)
for i in range(N_ret):
    day_idx = i + 1
    if day_idx >= SMA_LONG:
        sma200_dbc = np.mean(dbc_prices[day_idx - SMA_LONG:day_idx])
        if dbc_prices[day_idx] < sma200_dbc:
            ret_cmdty_trend[i] = 0.0
            continue
    if day_idx >= SMA_CMDTY:
        sma50 = np.mean(dbc_prices[day_idx - SMA_CMDTY:day_idx])
        deviation = (dbc_prices[day_idx] / sma50) - 1
        if deviation > 0:
            weight = min(deviation / 0.05, 1.0)
            ret_cmdty_trend[i] = ret["DBC"][i] * weight
        else:
            ret_cmdty_trend[i] = 0.0
    else:
        ret_cmdty_trend[i] = ret["DBC"][i] * 0.5

comp_ret_sma = {
    "Equity": ret_equity_sma, "Bonds": ret_bonds,
    "HardAssets": ret_hard_sma, "LongVol": ret_longvol,
    "CmdtyTrend": ret_cmdty_trend,
}

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

dragon_ret = monthly_rebal_portfolio(W_DRAGON, comp_ret_sma, dates_ret)

def cum_nav(returns):
    nav = np.ones(len(returns) + 1)
    for i, r in enumerate(returns):
        nav[i + 1] = nav[i] * (1 + r)
    return nav

nav_dragon = cum_nav(dragon_ret)

# ═══════════════════════════════════════════════════════════════════
# 6. CURRENT STATE
# ═══════════════════════════════════════════════════════════════════
print("Computing current state...")

last_entry = selection_log[-1]
last_rebal_date = last_entry["date"]

# Current SMA200 status for all tickers
current_sma = {}
for t in ALL_TICKERS:
    last_idx = N - 1
    sma_val = sma200_values[t][last_idx]
    price_val = price_data[t][last_idx]
    if not np.isnan(sma_val) and not np.isnan(price_val):
        current_sma[t] = {
            "price": price_val,
            "sma200": sma_val,
            "above": price_val > sma_val,
            "pct_vs_sma": ((price_val / sma_val) - 1) * 100,
        }

# Current exposure per block
current_exposure = {}
for block in W_DRAGON:
    block_key = block
    if block_key in exposure_scale and len(exposure_scale[block_key]) > 0:
        current_exposure[block] = exposure_scale[block_key][-1]
    else:
        current_exposure[block] = 1.0

# Current momentum scores
current_mom = {}
for t in ALL_TICKERS:
    if not np.isnan(mom[t][-1]):
        current_mom[t] = mom[t][-1] * 100

# YTD computation
ytd_start_idx = None
for i, d in enumerate(dates):
    if d.year == TODAY.year:
        ytd_start_idx = i
        break

ytd_data = {}
ytd_dates = []
ytd_dragon = np.array([0.0])
if ytd_start_idx is not None:
    ytd_dates = dates[ytd_start_idx:]
    for t in ALL_TICKERS:
        p = price_data[t]
        base = p[ytd_start_idx]
        if not np.isnan(base) and base > 0:
            ytd_data[t] = ((p[ytd_start_idx:] / base) - 1) * 100
    dragon_base = nav_dragon[ytd_start_idx]
    ytd_dragon = ((nav_dragon[ytd_start_idx:] / dragon_base) - 1) * 100

# Dragon YTD return
dragon_ytd = float(ytd_dragon[-1]) if len(ytd_dragon) > 0 else 0.0

# Dragon MTD return
mtd_start = None
for i, d in enumerate(dates):
    if d.year == TODAY.year and d.month == TODAY.month:
        mtd_start = i
        break
dragon_mtd = 0.0
if mtd_start is not None:
    dragon_mtd = ((nav_dragon[-1] / nav_dragon[mtd_start]) - 1) * 100

print(f"  Dragon YTD: {dragon_ytd:+.1f}%  MTD: {dragon_mtd:+.1f}%")
print(f"  Last rebalance: {last_rebal_date}")

# ═══════════════════════════════════════════════════════════════════
# 7. YTD CHART
# ═══════════════════════════════════════════════════════════════════
print("Generating charts...")

def build_ytd_chart():
    if not ytd_data or len(ytd_dates) < 2:
        return ''
    vw, vh = 720, 420
    ml, mr, mt, mb = 50, 90, 15, 30
    pw, ph = vw - ml - mr, vh - mt - mb
    n_pts = len(ytd_dates)
    all_vals = [0.0]
    for t in ytd_data:
        valid = ytd_data[t][~np.isnan(ytd_data[t])]
        if len(valid) > 0:
            all_vals.extend([float(np.min(valid)), float(np.max(valid))])
    all_vals.extend([float(np.min(ytd_dragon)), float(np.max(ytd_dragon))])
    y_min_raw, y_max_raw = min(all_vals), max(all_vals)
    pad = max(abs(y_max_raw - y_min_raw) * 0.08, 2)
    y_min, y_max = y_min_raw - pad, y_max_raw + pad
    y_range = y_max - y_min if y_max > y_min else 1
    svg = ""
    # Y grid
    span = y_max_raw - y_min_raw
    step = 2 if span < 15 else 5 if span < 40 else 10 if span < 80 else 20
    pct = int(math.floor(y_min / step)) * step
    while pct <= y_max + step:
        yp = mt + ph - ((pct - y_min) / y_range) * ph
        if yp < mt - 5 or yp > vh - mb + 5:
            pct += step
            continue
        w, op = ("1", "0.3") if pct == 0 else ("0.5", "0.12")
        svg += f'<line x1="{ml}" y1="{yp:.0f}" x2="{vw-mr}" y2="{yp:.0f}" stroke="rgba(148,163,184,{op})" stroke-width="{w}"/>'
        svg += f'<text x="{ml-5}" y="{yp:.0f}" text-anchor="end" fill="#64748b" font-size="8" dominant-baseline="middle">{pct:+.0f}%</text>'
        pct += step
    # X axis months
    month_names = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    seen = set()
    for i, d in enumerate(ytd_dates):
        if d.month not in seen:
            seen.add(d.month)
            x = ml + (i / max(n_pts - 1, 1)) * pw
            svg += f'<text x="{x:.0f}" y="{vh-5}" text-anchor="middle" fill="#64748b" font-size="8">{month_names[d.month-1]}</text>'
            svg += f'<line x1="{x:.0f}" y1="{mt}" x2="{x:.0f}" y2="{vh-mb}" stroke="rgba(148,163,184,0.06)" stroke-width="0.5"/>'
    # End values
    end_vals = {}
    for t in ALL_TICKERS:
        if t not in ytd_data: continue
        vals = ytd_data[t]
        for v in reversed(vals):
            if not np.isnan(v):
                end_vals[t] = float(v)
                break
    # Draw asset lines
    for t in end_vals:
        vals = ytd_data[t]
        color = TICKER_COLORS.get(t, "#94a3b8")
        pts = []
        for i in range(len(vals)):
            if not np.isnan(vals[i]):
                x = ml + (i / max(n_pts - 1, 1)) * pw
                y = mt + ph - ((vals[i] - y_min) / y_range) * ph
                pts.append(f"{x:.1f},{y:.1f}")
        if pts:
            svg += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1.3" stroke-linejoin="round" opacity="0.6"/>'
    # Dragon line (thick)
    pts = []
    for i in range(len(ytd_dragon)):
        x = ml + (i / max(n_pts - 1, 1)) * pw
        y = mt + ph - ((ytd_dragon[i] - y_min) / y_range) * ph
        pts.append(f"{x:.1f},{y:.1f}")
    svg += f'<polyline points="{" ".join(pts)}" fill="none" stroke="#06b6d4" stroke-width="2.5" stroke-linejoin="round"/>'
    # End labels (anti-overlap)
    all_labels = [(t, end_vals[t], TICKER_COLORS.get(t, "#94a3b8"), "7", "600") for t in end_vals]
    all_labels.append(("Centinela", float(ytd_dragon[-1]), "#06b6d4", "8", "800"))
    all_labels.sort(key=lambda x: -x[1])
    placed = []
    for name, val, color, fsize, fweight in all_labels:
        target_y = mt + ph - ((val - y_min) / y_range) * ph
        final_y = target_y
        for py in placed:
            if abs(final_y - py) < 9:
                final_y = py + 9
        placed.append(final_y)
        svg += f'<text x="{vw-mr+4}" y="{final_y:.0f}" fill="{color}" font-size="{fsize}" font-weight="{fweight}" dominant-baseline="middle">{name} {val:+.1f}%</text>'
    return f'<svg viewBox="0 0 {vw} {vh}" xmlns="http://www.w3.org/2000/svg">{svg}</svg>'

# ═══════════════════════════════════════════════════════════════════
# 8. POSITIONS TABLE
# ═══════════════════════════════════════════════════════════════════
def build_positions_html():
    rows = ""
    for block in ["Equity", "Bonds", "HardAssets", "LongVol", "CmdtyTrend"]:
        block_label = BLOCK_LABELS[block]
        block_color = BLOCK_COLORS[block]
        weight = W_DRAGON[block]
        exp = current_exposure.get(block, 1.0)

        candidates = UNIVERSES.get(block, UNIVERSES.get("Commodities", []))
        picks = last_entry.get(block, {}).get("picks", candidates) if block not in ["LongVol", "CmdtyTrend"] else candidates

        for j, t in enumerate(picks if block not in ["LongVol", "CmdtyTrend"] else candidates):
            sma_info = current_sma.get(t, {})
            above = sma_info.get("above", False)
            pct_vs = sma_info.get("pct_vs_sma", 0)
            price = sma_info.get("price", 0)
            sma_val = sma_info.get("sma200", 0)
            mom_val = current_mom.get(t, None)
            ytd_val = float(ytd_data[t][-1]) if t in ytd_data and len(ytd_data[t]) > 0 and not np.isnan(ytd_data[t][-1]) else None

            sma_icon = "▲" if above else "▼"
            sma_color = "#10b981" if above else "#ef4444"

            if j == 0:
                block_cell = f'<td rowspan="{len(picks) if block not in ["LongVol","CmdtyTrend"] else len(candidates)}" style="color:{block_color};font-weight:700;font-size:10px;vertical-align:top;border-right:2px solid {block_color}22;padding-right:12px">{block_label}<br><span style="font-size:8px;color:#64748b;font-weight:400">{int(weight*100)}% | Exp {exp*100:.0f}%</span></td>'
            else:
                block_cell = ""

            rows += f'''<tr>
                {block_cell}
                <td style="font-weight:700;color:{TICKER_COLORS.get(t,'#94a3b8')}">{t}</td>
                <td style="color:#64748b;font-size:9px">{TICKER_LABELS.get(t, t)}</td>
                <td class="num">${price:,.2f}</td>
                <td class="num" style="color:#64748b">${sma_val:,.2f}</td>
                <td class="num" style="color:{sma_color};font-weight:700">{sma_icon} {pct_vs:+.1f}%</td>
                <td class="num" style="color:{'#10b981' if mom_val and mom_val > 0 else '#ef4444' if mom_val and mom_val < 0 else '#64748b'}">{f'{mom_val:+.1f}%' if mom_val is not None else '--'}</td>
                <td class="num" style="font-weight:600;color:{'#10b981' if ytd_val and ytd_val > 0 else '#ef4444' if ytd_val and ytd_val < 0 else '#64748b'}">{f'{ytd_val:+.1f}%' if ytd_val is not None else '--'}</td>
            </tr>'''
    return rows

# ═══════════════════════════════════════════════════════════════════
# 9. EXPOSURE BARS
# ═══════════════════════════════════════════════════════════════════
def build_exposure_bars():
    bars = ""
    for block in ["Equity", "Bonds", "HardAssets", "LongVol", "CmdtyTrend"]:
        label = BLOCK_LABELS[block]
        color = BLOCK_COLORS[block]
        exp = current_exposure.get(block, 1.0)
        weight = W_DRAGON[block]
        effective = weight * exp
        picks = last_entry.get(block, {}).get("picks", UNIVERSES.get(block, UNIVERSES.get("Commodities", [])))

        # Count above SMA200
        above_count = sum(1 for t in picks if current_sma.get(t, {}).get("above", False))
        total_picks = len(picks)

        bars += f'''<div style="margin-bottom:12px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
                <span style="font-size:10px;font-weight:700;color:{color}">{label}</span>
                <span style="font-size:9px;color:#94a3b8">{above_count}/{total_picks} > SMA200 | Exp: {exp*100:.0f}% | Efectivo: {effective*100:.1f}%</span>
            </div>
            <div style="height:20px;background:rgba(148,163,184,0.06);border-radius:4px;overflow:hidden;position:relative">
                <div style="width:{exp*100:.0f}%;height:100%;background:{color};opacity:0.4;border-radius:4px"></div>
                <div style="position:absolute;top:0;left:0;width:{weight*100:.0f}%;height:100%;border-right:2px dashed {color};opacity:0.6"></div>
            </div>
        </div>'''
    return bars

# ═══════════════════════════════════════════════════════════════════
# 10. HTML REPORT
# ═══════════════════════════════════════════════════════════════════
print("Generating HTML report...")

pct_cls = lambda v: "pos" if v > 0 else "neg" if v < 0 else ""

html = f'''<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Centinela v3 — Live Dashboard | SFinance</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: 'Inter', -apple-system, sans-serif; background:#0f172a; color:#e2e8f0; }}
.container {{ max-width:1200px; margin:0 auto; padding:24px 20px; }}
.header {{ display:flex; justify-content:space-between; align-items:center; padding:16px 0; border-bottom:1px solid rgba(148,163,184,0.12); margin-bottom:24px; }}
.header-title {{ font-size:22px; font-weight:800; letter-spacing:-0.5px; }}
.header-title span {{ color:#06b6d4; }}
.header-sub {{ font-size:11px; color:#64748b; text-align:right; }}
.header-sub strong {{ color:#94a3b8; }}
.pos {{ color:#10b981; }} .neg {{ color:#ef4444; }}
.kpi-strip {{ display:grid; grid-template-columns:repeat(5, 1fr); gap:10px; margin-bottom:24px; }}
.kpi {{ background:rgba(30,41,59,0.6); border:1px solid rgba(148,163,184,0.08); border-radius:8px; padding:16px 12px; text-align:center; }}
.kpi-label {{ font-size:8px; text-transform:uppercase; color:#64748b; letter-spacing:0.5px; margin-bottom:6px; }}
.kpi-value {{ font-size:22px; font-weight:800; letter-spacing:-0.5px; }}
.kpi-sub {{ font-size:9px; color:#475569; margin-top:4px; }}
.section {{ margin-bottom:24px; }}
.section-title {{ font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; color:#94a3b8; margin-bottom:12px; padding-left:2px; }}
.chart-container {{ background:rgba(15,23,42,0.5); border:1px solid rgba(148,163,184,0.06); border-radius:8px; padding:16px; }}
.card {{ background:rgba(30,41,59,0.4); border:1px solid rgba(148,163,184,0.08); border-radius:8px; padding:16px; }}
.grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:24px; }}
.data-table {{ width:100%; border-collapse:collapse; font-size:10px; }}
.data-table th {{ background:rgba(30,41,59,0.8); color:#94a3b8; font-weight:600; text-transform:uppercase; font-size:8px; letter-spacing:0.5px; padding:8px 6px; text-align:left; border-bottom:1px solid rgba(148,163,184,0.12); }}
.data-table td {{ padding:7px 6px; border-bottom:1px solid rgba(148,163,184,0.06); }}
.data-table .num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.data-table tr:hover {{ background:rgba(148,163,184,0.04); }}
.live-dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; background:#22c55e; margin-right:6px; animation:pulse 2s infinite; }}
@keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.4; }} }}
.footer {{ display:flex; justify-content:space-between; font-size:8px; color:#475569; padding:16px 0; border-top:1px solid rgba(148,163,184,0.08); margin-top:20px; }}
@media (max-width:900px) {{ .kpi-strip {{ grid-template-columns:repeat(3, 1fr); }} .grid-2 {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <div>
      <div class="header-title"><span class="live-dot"></span><span>Centinela v3</span> — Live Dashboard</div>
      <div style="font-size:10px;color:#64748b;margin-top:2px">SMA200 Trend Filter | Top-{N_SELECT} Momentum {MOM_LOOKBACK}d | {len(ALL_TICKERS)} activos</div>
    </div>
    <div class="header-sub"><strong>SFinance-alicIA</strong><br>Actualizado: {TODAY.strftime("%d %b %Y")}<br>Ultimo rebalanceo: {last_rebal_date.strftime("%d %b %Y")}</div>
  </div>

  <div class="kpi-strip">
    <div class="kpi" style="border-top:3px solid #06b6d4">
      <div class="kpi-label">YTD {TODAY.year}</div>
      <div class="kpi-value {pct_cls(dragon_ytd)}">{dragon_ytd:+.1f}%</div>
      <div class="kpi-sub">Centinela v3 SMA200</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">MTD {TODAY.strftime("%b")}</div>
      <div class="kpi-value {pct_cls(dragon_mtd)}">{dragon_mtd:+.1f}%</div>
      <div class="kpi-sub">Mes en curso</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">NAV</div>
      <div class="kpi-value" style="color:#06b6d4">${nav_dragon[-1]:.2f}</div>
      <div class="kpi-sub">Desde $1 ({dates[0]})</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Activos > SMA200</div>
      <div class="kpi-value">{sum(1 for t in current_sma if current_sma[t]['above'])}/{len(current_sma)}</div>
      <div class="kpi-sub">Del universo total</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Ultimo Dato</div>
      <div class="kpi-value" style="font-size:14px;color:#94a3b8">{dates[-1].strftime("%d %b")}</div>
      <div class="kpi-sub">{N_ret} trading days</div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Evolutivo YTD {TODAY.year} — Todos los Activos (Base 0%)</div>
    <div class="chart-container">{build_ytd_chart()}</div>
    <div style="margin-top:8px;font-size:8px;color:#475569">
      Retorno acumulado desde el 1 de enero {TODAY.year}. Cada linea representa un activo individual del universo.
      <span style="color:#06b6d4;font-weight:700">Linea cyan = Centinela v3 SMA200.</span>
      Etiquetas a la derecha muestran el retorno YTD actual.
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="section-title">Exposicion por Bloque — SMA200 Filter</div>
      {build_exposure_bars()}
      <div style="font-size:8px;color:#475569;margin-top:8px">
        Barra = exposicion actual (% del target). Linea punteada = peso target.
        Min exposure: {MIN_EXPOSURE*100:.0f}%. Cash proxy: SHY.
      </div>
    </div>
    <div class="card">
      <div class="section-title">Senales del Ultimo Rebalanceo — {last_rebal_date.strftime("%b %Y")}</div>
      <div style="font-size:9px;color:#94a3b8;line-height:1.8">
        {''.join(
            f'<div style="margin-bottom:8px">'
            f'<span style="color:{BLOCK_COLORS[block]};font-weight:700">{BLOCK_LABELS[block]}</span>: '
            + ', '.join(
                f'<span style="color:{TICKER_COLORS.get(t,"#94a3b8")};font-weight:600">{t}</span>'
                + (f' <span style="color:#10b981;font-size:8px">▲</span>' if current_sma.get(t,{}).get("above",False) else f' <span style="color:#ef4444;font-size:8px">▼</span>')
                for t in last_entry.get(block, {}).get("picks", UNIVERSES.get(block, UNIVERSES.get("Commodities", [])))
            )
            + f'</div>'
            for block in ["Equity", "Bonds", "HardAssets", "LongVol", "CmdtyTrend"]
        )}
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Posiciones Actuales — Detalle</div>
    <div style="overflow-x:auto">
    <table class="data-table">
      <tr>
        <th>Bloque</th><th>Ticker</th><th>Activo</th>
        <th class="num">Precio</th><th class="num">SMA200</th>
        <th class="num">vs SMA200</th><th class="num">Mom 6M</th><th class="num">YTD</th>
      </tr>
      {build_positions_html()}
    </table>
    </div>
  </div>

  <div class="footer">
    <span>SFinance-alicIA | Centinela v3 SMA200 — Live Dashboard | Solo fines informativos</span>
    <span>{TODAY.strftime("%Y-%m-%d")} | Actualizacion diaria 22:00 UTC</span>
  </div>

</div>
</body>
</html>'''

# ═══════════════════════════════════════════════════════════════════
# 11. OUTPUT
# ═══════════════════════════════════════════════════════════════════
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dragon_Live.html")
with open(outpath, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\n  Report saved: {outpath}")

if not os.environ.get("CI"):
    os.system(f'open "{outpath}"')
print("\n=== Done ===")
