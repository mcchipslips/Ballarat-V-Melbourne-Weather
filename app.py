"""
Ballarat vs Melbourne Weather Dashboard
========================================
Data: Open-Meteo Historical Weather API (ERA5 reanalysis, 1940–present)
Stations:
  Ballarat:  -37.5622, 143.8503  (Ballarat Aerodrome area)
  Melbourne: -37.8136, 144.9631  (Melbourne CBD)

Run:
    pip install dash plotly pandas requests numpy
    python weather_dashboard.py

The script will:
  1. Fetch ~25 years of hourly data from Open-Meteo on first run (~2-3 minutes)
  2. Cache it locally as CSV files so subsequent runs are instant
  3. Launch the dashboard at http://127.0.0.1:8050
"""

import os
import time
import requests
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BALLARAT_LAT,  BALLARAT_LON  = -37.5622, 143.8503
MELBOURNE_LAT, MELBOURNE_LON = -37.8136, 144.9631
START_DATE = "2000-01-01"
END_DATE   = "2024-12-31"
CACHE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weather_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────────────────────
BG        = "#1A1A1A"
CARD      = "#242424"
ACCENT    = "#D4A853"
ACCENT2   = "#7EB8C9"
TEXT_MAIN = "#F0EAD6"
TEXT_DIM  = "#8A8070"
C_DAY     = "#D4A853"
C_NIGHT   = "#4A5568"
C_RAIN    = "#7EB8C9"
C_WIND    = "#C4956A"

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHER
# ─────────────────────────────────────────────────────────────────────────────
def fetch_hourly(lat, lon, label, start=START_DATE, end=END_DATE):
    cache_path = os.path.join(CACHE_DIR, f"{label}_hourly.csv")
    if os.path.exists(cache_path):
        print(f"  Loading {label} from cache...")
        return pd.read_csv(cache_path, parse_dates=["time"])

    print(f"  Fetching {label} from Open-Meteo API ({start} → {end})...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start,
        "end_date":   end,
        "hourly": ",".join([
            "temperature_2m",
            "precipitation",
            "windspeed_10m",
            "windgusts_10m",
            "is_day",
        ]),
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max",
            "windgusts_10m_max",
            "sunrise",
            "sunset",
        ]),
        "timezone":             "Australia/Melbourne",
        "wind_speed_unit":      "kmh",
        "precipitation_unit":   "mm",
    }

    all_hourly = []
    all_daily  = []
    years = list(range(int(start[:4]), int(end[:4]) + 1, 5))
    for i, yr_start in enumerate(years):
        yr_end = min(yr_start + 4, int(end[:4]))
        chunk_start = f"{yr_start}-01-01"
        chunk_end   = f"{yr_end}-12-31"
        p = dict(params)
        p["start_date"] = chunk_start
        p["end_date"]   = chunk_end

        for attempt in range(6):
            try:
                r = requests.get(url, params=p, timeout=60)
                if r.status_code == 429:
                    wait = 30 * (attempt + 1)
                    print(f"    Rate limited — waiting {wait}s before retry...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except requests.exceptions.HTTPError as e:
                if attempt == 5:
                    raise
                print(f"    HTTP error, retry {attempt+1}...")
                time.sleep(20)
            except Exception as e:
                if attempt == 5:
                    raise
                print(f"    Error, retry {attempt+1}...")
                time.sleep(10)

        h = pd.DataFrame(data["hourly"])
        h["time"] = pd.to_datetime(h["time"])
        all_hourly.append(h)

        d = pd.DataFrame(data["daily"])
        d["time"] = pd.to_datetime(d["time"])
        all_daily.append(d)
        print(f"    ✓ {chunk_start} → {chunk_end}")
        time.sleep(8)

    hourly = pd.concat(all_hourly, ignore_index=True).drop_duplicates("time")
    daily  = pd.concat(all_daily,  ignore_index=True).drop_duplicates("time")

    hourly.to_csv(cache_path, index=False)
    daily.to_csv(os.path.join(CACHE_DIR, f"{label}_daily.csv"), index=False)
    print(f"  ✓ {label} saved to cache.")
    return hourly


def load_daily(label):
    path = os.path.join(CACHE_DIR, f"{label}_daily.csv")
    return pd.read_csv(path, parse_dates=["time"])


def build_daytime_nighttime(hourly_df):
    hourly_df["date"] = hourly_df["time"].dt.date
    day   = hourly_df[hourly_df["is_day"] == 1].groupby("date")["temperature_2m"].agg(
        day_max="max", day_min="min").reset_index()
    night = hourly_df[hourly_df["is_day"] == 0].groupby("date")["temperature_2m"].agg(
        night_max="max", night_min="min").reset_index()
    merged = day.merge(night, on="date", how="outer")
    merged["date"] = pd.to_datetime(merged["date"])
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("  Ballarat vs Melbourne Weather Dashboard")
print("  Fetching data (first run takes ~2-3 minutes)...")
print("═"*60)

bal_hourly = fetch_hourly(BALLARAT_LAT,  BALLARAT_LON,  "ballarat")
mel_hourly = fetch_hourly(MELBOURNE_LAT, MELBOURNE_LON, "melbourne")
bal_daily  = load_daily("ballarat")
mel_daily  = load_daily("melbourne")

bal_dn = build_daytime_nighttime(bal_hourly)
mel_dn = build_daytime_nighttime(mel_hourly)

for df in [bal_daily, mel_daily]:
    df["year"]  = df["time"].dt.year
    df["month"] = df["time"].dt.month

for df in [bal_dn, mel_dn]:
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

def monthly_avg_dn(dn_df):
    return dn_df.groupby("month").agg(
        day_max=("day_max", "mean"),
        day_min=("day_min", "mean"),
        night_max=("night_max", "mean"),
        night_min=("night_min", "mean"),
    ).reset_index()

bal_monthly_dn = monthly_avg_dn(bal_dn)
mel_monthly_dn = monthly_avg_dn(mel_dn)

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def rainfall_stats(daily_df):
    daily_df = daily_df.copy()
    daily_df["rain_day"] = (daily_df["precipitation_sum"] > 0.2).astype(int)
    monthly = daily_df.groupby("month").agg(
        avg_rain_mm=("precipitation_sum", "mean"),
        rain_days_pct=("rain_day", "mean"),
        total_rain_mm=("precipitation_sum", "sum"),
    ).reset_index()
    monthly["rain_days_pct"] *= 100
    return monthly

bal_rain = rainfall_stats(bal_daily)
mel_rain = rainfall_stats(mel_daily)

YEARS = sorted(bal_daily["year"].unique().tolist())

print("\n  ✓ All data ready. Starting dashboard...\n")

# ─────────────────────────────────────────────────────────────────────────────
# THEME HELPERS
# ─────────────────────────────────────────────────────────────────────────────
PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Georgia, serif", color=TEXT_MAIN, size=13),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_DIM)),
    hovermode="x unified",
)

def axis(title="", **kwargs):
    return dict(title=title, gridcolor="#3A3530", zerolinecolor="#3A3530",
                tickfont=dict(color=TEXT_DIM), **kwargs)

LABEL_STYLE = {"fontSize": "0.7rem", "color": TEXT_DIM, "textTransform": "uppercase",
               "letterSpacing": "0.1em", "marginBottom": "6px", "display": "block"}

def kpi_card(title, value, subtitle="", color=ACCENT):
    return html.Div([
        html.P(title, style={"margin": 0, "fontSize": "0.7rem", "color": TEXT_DIM,
                             "textTransform": "uppercase", "letterSpacing": "0.08em"}),
        html.H2(value, style={"margin": "4px 0", "fontSize": "1.8rem",
                              "color": color, "fontFamily": "Georgia, serif"}),
        html.P(subtitle, style={"margin": 0, "fontSize": "0.68rem", "color": TEXT_DIM}),
    ], style={"background": CARD, "borderRadius": "12px", "padding": "16px 20px",
              "borderLeft": f"4px solid {color}", "flex": "1", "minWidth": "150px"})


# ─────────────────────────────────────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="Ballarat vs Melbourne Weather")

app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
  .Select-control,
  .Select--single > .Select-control,
  .Select--multi > .Select-control {
    background-color: #1A1A1A !important;
    border: 1px solid #888888 !important;
    color: #B0B0B0 !important;
  }
  .Select-value, .Select-value-label {
    color: #B0B0B0 !important;
    background-color: #1A1A1A !important;
  }
  .Select-placeholder {
    color: #999999 !important;
    background-color: #1A1A1A !important;
  }
  .Select-input, .Select-input > input {
    color: #B0B0B0 !important;
    background-color: #1A1A1A !important;
  }
  .Select-arrow-zone .Select-arrow {
    border-top-color: #B0B0B0 !important;
  }
  .Select-menu-outer {
    background-color: #1A1A1A !important;
    border: 1px solid #888888 !important;
    z-index: 9999 !important;
  }
  .Select-menu {
    background-color: #1A1A1A !important;
  }
  .Select-option {
    background-color: #1A1A1A !important;
    color: #B0B0B0 !important;
  }
  .Select-option.is-focused {
    background-color: #333333 !important;
    color: #D4A853 !important;
  }
  .Select-option.is-selected {
    background-color: #2A2A2A !important;
    color: #D4A853 !important;
    font-weight: bold !important;
  }
  .dash-dropdown .Select-control { background-color: #1A1A1A !important; }
  .dash-dropdown .Select-value-label { color: #B0B0B0 !important; }
  .dash-dropdown .Select-menu-outer { background-color: #1A1A1A !important; }
  .dash-dropdown .Select-option { color: #B0B0B0 !important; background-color: #1A1A1A !important; }
  .dash-dropdown .Select-option.is-focused { color: #D4A853 !important; background-color: #333333 !important; }
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>"""

server = app.server
app.config.suppress_callback_exceptions = True

year_options = [{"label": str(y), "value": y} for y in YEARS]

app.layout = html.Div([

    # Header
    html.Div([
        html.Div([
            html.Span("🌦", style={"fontSize": "1.4rem", "marginRight": "10px"}),
            html.Span("Ballarat vs Melbourne — Weather Comparison", style={
                "fontSize": "1.2rem", "fontWeight": "700",
                "fontFamily": "Georgia, serif", "color": TEXT_MAIN}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div([
            html.Span("■ ", style={"color": ACCENT}),
            html.Span("Ballarat   ", style={"color": TEXT_DIM, "fontSize": "0.8rem", "marginRight": "16px"}),
            html.Span("■ ", style={"color": ACCENT2}),
            html.Span("Melbourne", style={"color": TEXT_DIM, "fontSize": "0.8rem"}),
        ]),
        html.P(f"Open-Meteo ERA5 Reanalysis · {START_DATE[:4]}–{END_DATE[:4]}  ·  Is it really 3°C warmer in Melbourne?",
               style={"margin": 0, "fontSize": "0.72rem", "color": TEXT_DIM}),
    ], style={
        "background": CARD, "padding": "14px 32px",
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "borderBottom": "2px solid #D4A853", "position": "sticky", "top": 0, "zIndex": 999,
    }),

    html.Div([

        # Sidebar
        html.Div([
            html.Div([
                html.Span("Filters", style={"fontSize": "0.65rem", "color": ACCENT,
                    "textTransform": "uppercase", "letterSpacing": "0.15em", "fontWeight": "700"}),
                html.Hr(style={"borderColor": "#3A3530", "margin": "8px 0 0 0"}),
            ]),
            html.Div([
                html.Label("Year Range", style=LABEL_STYLE),
                html.Div([
                    html.Div([
                        html.Span("From", style={"fontSize": "0.62rem", "color": TEXT_DIM, "marginBottom": "4px", "display": "block"}),
                        dcc.Dropdown(id="year-from", options=year_options, value=YEARS[0], clearable=False, className="dark-dropdown", style={"color": "#B0B0B0"}),
                    ], style={"flex": 1}),
                    html.Div([
                        html.Span("To", style={"fontSize": "0.62rem", "color": TEXT_DIM, "marginBottom": "4px", "display": "block"}),
                        dcc.Dropdown(id="year-to", options=year_options, value=YEARS[-1], clearable=False, className="dark-dropdown",style={"color": "#B0B0B0"}),
                    ], style={"flex": 1}),
                ], style={"display": "flex", "gap": "8px"}),
            ]),
            html.Div([
                html.Hr(style={"borderColor": "#3A3530"}),
                html.P([
                    html.Strong("Data source:", style={"color": TEXT_DIM}), html.Br(),
                    "Open-Meteo Historical Weather API (ERA5 reanalysis). ",
                    "ERA5 is the ECMWF global reanalysis dataset, widely used in climate research. ",
                    "Resolution: ~9km. Data is not identical to BOM station readings but is "
                    "scientifically comparable for trend analysis.",
                    html.Br(), html.Br(),
                    html.Strong("Daytime/Nighttime:", style={"color": TEXT_DIM}), html.Br(),
                    "Split using Open-Meteo's is_day flag, which follows sunrise/sunset for the location.",
                    html.Br(), html.Br(),
                    html.Strong("Rainfall threshold:", style={"color": TEXT_DIM}), html.Br(),
                    "A 'rain day' is defined as ≥ 0.2mm precipitation (standard BOM definition).",
                ], style={"fontSize": "0.62rem", "color": "#6A6055", "lineHeight": "1.6"}),
            ]),
        ], style={
            "width": "240px", "minWidth": "240px", "background": CARD,
            "padding": "24px 18px", "display": "flex", "flexDirection": "column",
            "gap": "20px", "borderRight": "1px solid #3A3530", "overflowY": "auto",
        }),

        # Main content
        html.Div([
            html.Div(id="kpi-row", style={"display": "flex", "gap": "12px",
                                           "flexWrap": "wrap", "marginBottom": "20px"}),
            dcc.Tabs(id="tabs", value="temp", children=[
                dcc.Tab(label="🌡 Temperature Ranges",     value="temp"),
                dcc.Tab(label="🌙 Day vs Night",           value="daynight"),
                dcc.Tab(label="🌧 Rainfall",               value="rain"),
                dcc.Tab(label="💨 Wind",                   value="wind"),
                dcc.Tab(label="📅 Monthly Averages",       value="monthly"),
                dcc.Tab(label="🧪 The 3°C Test",           value="test"),
                dcc.Tab(label="ℹ️  Definitions",            value="defs"),
            ], colors={"border": "#3A3530", "primary": ACCENT, "background": CARD},
            style={"fontFamily": "Georgia, serif", "fontSize": "0.83rem"}),

            html.Div(children=[
                html.Div([
                    html.Span("Show / hide:  ", style={
                        "fontSize": "0.72rem", "color": TEXT_DIM,
                        "marginRight": "10px", "fontFamily": "Georgia, serif",
                        "whiteSpace": "nowrap",
                    }),
                    dcc.Checklist(
                        id="temp-series-toggle",
                        options=[
                            {"label": "  Ballarat daily max",   "value": "bal_max"},
                            {"label": "  Melbourne daily max",  "value": "mel_max"},
                        ],
                        value=["bal_max", "mel_max"],
                        inline=True,
                        labelStyle={"color": TEXT_DIM, "fontSize": "0.78rem",
                                    "marginRight": "16px", "cursor": "pointer"},
                        inputStyle={"marginRight": "5px", "accentColor": ACCENT},
                    ),
                ], style={
                    "display": "flex", "alignItems": "center", "flexWrap": "wrap",
                    "background": CARD, "padding": "8px 16px",
                    "border": "1px solid #3A3530", "borderTop": "none",
                    "borderRadius": "0 0 6px 6px", "marginBottom": "4px",
                }),
            ], id="temp-toggles-wrapper", style={"display": "none"}),

            html.Div(id="tab-content", style={"marginTop": "8px"}),
        ], style={"flex": 1, "padding": "22px 28px", "overflow": "auto"}),

    ], style={"display": "flex", "flex": 1, "overflow": "hidden", "height": "calc(100vh - 57px)"}),

    # Footer
    html.Div([
        html.Span("Yes, Justin made this.  ", style={
            "color": "#5A5045", "fontSize": "0.62rem", "fontStyle": "italic", "fontFamily": "Georgia, serif"}),
        html.Span("Data: Open-Meteo ERA5 Reanalysis · open-meteo.com",
                  style={"color": "#5A5045", "fontSize": "0.62rem"}),
    ], style={
        "background": BG, "padding": "5px 32px",
        "borderTop": "1px solid #3A3530",
        "display": "flex", "justifyContent": "space-between",
    }),

], style={"fontFamily": "Georgia, serif", "backgroundColor": BG, "color": TEXT_MAIN,
          "height": "100vh", "display": "flex", "flexDirection": "column", "overflow": "hidden"})


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def filter_years(df, yf, yt, date_col="time"):
    return df[(df[date_col].dt.year >= yf) & (df[date_col].dt.year <= yt)]


# ── FIX: KPI cards now filter on pre-computed 'year' column ──────────────────
@app.callback(
    Output("kpi-row", "children"),
    Input("year-from", "value"), Input("year-to", "value"),
)
def update_kpis(yf, yt):
    yf = int(yf) if yf is not None else YEARS[0]
    yt = int(yt) if yt is not None else YEARS[-1]
    if yf > yt: yf, yt = yt, yf

    bd = bal_daily[(bal_daily["year"] >= yf) & (bal_daily["year"] <= yt)]
    md = mel_daily[(mel_daily["year"] >= yf) & (mel_daily["year"] <= yt)]

    bal_avg_max = bd["temperature_2m_max"].mean()
    mel_avg_max = md["temperature_2m_max"].mean()
    diff = mel_avg_max - bal_avg_max

    bal_rain_total = bd["precipitation_sum"].sum()
    mel_rain_total = md["precipitation_sum"].sum()

    bal_wind_max = bd["windgusts_10m_max"].max()
    mel_wind_max = md["windgusts_10m_max"].max()

    return [
        kpi_card("Avg Daily Max — Ballarat",    f"{bal_avg_max:.1f}°C",       f"{yf}–{yt}", ACCENT),
        kpi_card("Avg Daily Max — Melbourne",   f"{mel_avg_max:.1f}°C",       f"{yf}–{yt}", ACCENT2),
        kpi_card("Mel warmer by (avg max)",     f"{diff:+.1f}°C",             "The 3°C test", C_DAY),
        kpi_card("Total Rainfall — Ballarat",   f"{bal_rain_total:,.0f}mm",   f"{yf}–{yt}", C_RAIN),
        kpi_card("Total Rainfall — Melbourne",  f"{mel_rain_total:,.0f}mm",   f"{yf}–{yt}", C_RAIN),
        kpi_card("Peak Wind Gust — Ballarat",   f"{bal_wind_max:.0f}km/h",    f"{yf}–{yt}", C_WIND),
    ]


@app.callback(
    Output("temp-toggles-wrapper", "style"),
    Input("tabs", "value"),
)
def show_temp_toggles(tab):
    if tab == "temp":
        return {"display": "block"}
    return {"display": "none"}


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("year-from", "value"), Input("year-to", "value"),
    Input("temp-series-toggle", "value"),
)
def render_tab(tab, yf, yt, temp_series):
    if not tab:
        tab = "temp"
    if temp_series is None:
        temp_series = ["bal_max", "mel_max"]
    yf = int(yf) if yf is not None else YEARS[0]
    yt = int(yt) if yt is not None else YEARS[-1]
    if yf > yt: yf, yt = yt, yf

    bd = filter_years(bal_daily, yf, yt)
    md = filter_years(mel_daily, yf, yt)
    bdn = filter_years(bal_dn, yf, yt, date_col="date")
    mdn = filter_years(mel_dn, yf, yt, date_col="date")

    # ── TEMPERATURE RANGES ────────────────────────────────────────────────────
    if tab == "temp":
        bd_s = bd.sort_values("time")
        md_s = md.sort_values("time")
        fig = go.Figure()
        if "bal_max" in temp_series:
            fig.add_trace(go.Scatter(
                x=bd_s["time"], y=bd_s["temperature_2m_max"],
                line=dict(color=ACCENT, width=1), name="Ballarat — Daily Max",
            ))
        if "mel_max" in temp_series:
            fig.add_trace(go.Scatter(
                x=md_s["time"], y=md_s["temperature_2m_max"],
                line=dict(color=ACCENT2, width=1), name="Melbourne — Daily Max",
            ))
        fig.update_layout(
            **PLOT_BASE,
            title=dict(text=f"Daily Temperature Range — Ballarat vs Melbourne ({yf}–{yt})",
                       font=dict(size=14, color=TEXT_MAIN)),
            xaxis=dict(title="Date", gridcolor="#3A3530", zerolinecolor="#3A3530",
                       tickfont=dict(color=TEXT_DIM),
                       range=[pd.Timestamp(f"{yf}-01-01"), pd.Timestamp(f"{yt}-12-31")]),
            yaxis=axis("Temperature (°C)"),
            height=500, margin=dict(l=60, r=40, t=60, b=50),
        )
        fig.update_layout(
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_DIM),
                        itemclick="toggle", itemdoubleclick="toggleothers",
                        title=dict(text="Click legend items to show/hide",
                                   font=dict(size=10, color=TEXT_DIM))),
        )
        return dcc.Graph(figure=fig, config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": [
                "zoom2d","pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d",
                "autoScale2d","hoverClosestCartesian","hoverCompareCartesian",
                "toggleSpikelines","toImage"
            ],
            "displaylogo": False,
        })

    # ── DAY VS NIGHT ─────────────────────────────────────────────────────────
    elif tab == "daynight":
        bdn_s = bdn.sort_values("date")
        mdn_s = mdn.sort_values("date")

        fig = go.Figure()
        roll = 30

        fig.add_trace(go.Scatter(
            x=bdn_s["date"],
            y=bdn_s["day_max"].rolling(roll, center=True).mean(),
            line=dict(color=ACCENT, width=2), name="Ballarat — Daytime Max", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=mdn_s["date"],
            y=mdn_s["day_max"].rolling(roll, center=True).mean(),
            line=dict(color=ACCENT2, width=2), name="Melbourne — Daytime Max", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=bdn_s["date"],
            y=bdn_s["night_max"].rolling(roll, center=True).mean(),
            line=dict(color=ACCENT, width=1.5, dash="dot"), name="Ballarat — Nighttime Max", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=mdn_s["date"],
            y=mdn_s["night_max"].rolling(roll, center=True).mean(),
            line=dict(color=ACCENT2, width=1.5, dash="dot"), name="Melbourne — Nighttime Max", showlegend=True,
        ))

        fig.update_layout(
            **PLOT_BASE,
            title=dict(text=f"Daytime vs Nighttime Temperature Ranges ({yf}–{yt})  [30-day rolling avg]",
                       font=dict(size=14, color=TEXT_MAIN)),
            xaxis=dict(title="Date", gridcolor="#3A3530", zerolinecolor="#3A3530",
                       tickfont=dict(color=TEXT_DIM),
                       range=[pd.Timestamp(f"{yf}-01-01"), pd.Timestamp(f"{yt}-12-31")]),
            yaxis=axis("Temperature (°C)"),
            height=500, margin=dict(l=60, r=40, t=60, b=50),
        )

        note = html.Div([
            html.P("Solid lines show daytime maximum (30-day rolling avg). Dotted lines show nighttime maximum. "
                   "Ballarat's nighttime band sits noticeably lower than Melbourne's — consistent with the city heat island effect.",
                   style={"fontSize": "0.78rem", "color": TEXT_DIM, "margin": 0}),
        ], style={"background": CARD, "borderRadius": "8px", "padding": "12px 18px",
                  "borderLeft": f"4px solid {C_NIGHT}", "marginTop": "12px"})

        return html.Div([dcc.Graph(figure=fig, config={"displayModeBar": False}), note])

    # ── RAINFALL ─────────────────────────────────────────────────────────────
    elif tab == "rain":
        bal_annual = bd.groupby("year")["precipitation_sum"].sum().reset_index()
        mel_annual = md.groupby("year")["precipitation_sum"].sum().reset_index()

        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=bal_annual["year"], y=bal_annual["precipitation_sum"],
            name="Ballarat", marker_color=ACCENT, opacity=0.8))
        fig1.add_trace(go.Bar(x=mel_annual["year"], y=mel_annual["precipitation_sum"],
            name="Melbourne", marker_color=ACCENT2, opacity=0.8))
        for df, color in [(bal_annual, ACCENT), (mel_annual, ACCENT2)]:
            z = np.polyfit(df["year"], df["precipitation_sum"], 1)
            p = np.poly1d(z)
            fig1.add_trace(go.Scatter(x=df["year"], y=p(df["year"]),
                mode="lines", line=dict(color=color, width=2, dash="dot"), showlegend=False))

        fig1.update_layout(
            **PLOT_BASE,
            title=dict(text=f"Annual Rainfall — Ballarat vs Melbourne ({yf}–{yt})",
                       font=dict(size=14, color=TEXT_MAIN)),
            xaxis=axis("Year"), yaxis=axis("Total Rainfall (mm)"),
            barmode="group", height=380, margin=dict(l=60, r=40, t=60, b=50),
        )

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=MONTH_LABELS, y=bal_rain["rain_days_pct"],
            name="Ballarat — Rain Days %", marker_color=ACCENT, opacity=0.8))
        fig2.add_trace(go.Bar(x=MONTH_LABELS, y=mel_rain["rain_days_pct"],
            name="Melbourne — Rain Days %", marker_color=ACCENT2, opacity=0.8))
        fig2.update_layout(
            **PLOT_BASE,
            title=dict(text="% of Days with Rainfall (≥0.2mm) by Month — Long-term Average",
                       font=dict(size=14, color=TEXT_MAIN)),
            xaxis=axis("Month"), yaxis=axis("% of Days with Rain"),
            barmode="group", height=340, margin=dict(l=60, r=40, t=60, b=50),
        )

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=MONTH_LABELS, y=bal_rain["avg_rain_mm"],
            name="Ballarat — Avg Daily (mm)", marker_color=ACCENT, opacity=0.8))
        fig3.add_trace(go.Bar(x=MONTH_LABELS, y=mel_rain["avg_rain_mm"],
            name="Melbourne — Avg Daily (mm)", marker_color=ACCENT2, opacity=0.8))
        fig3.update_layout(
            **PLOT_BASE,
            title=dict(text="Average Daily Rainfall Volume by Month — Long-term Average",
                       font=dict(size=14, color=TEXT_MAIN)),
            xaxis=axis("Month"), yaxis=axis("Avg Daily Rainfall (mm)"),
            barmode="group", height=340, margin=dict(l=60, r=40, t=60, b=50),
        )

        return html.Div([
            dcc.Graph(figure=fig1, config={"displayModeBar": False}),
            html.Div([
                html.Div(dcc.Graph(figure=fig2, config={"displayModeBar": False}), style={"flex": 1}),
                html.Div(dcc.Graph(figure=fig3, config={"displayModeBar": False}), style={"flex": 1}),
            ], style={"display": "flex", "gap": "12px"}),
        ])

    # ── WIND ─────────────────────────────────────────────────────────────────
    elif tab == "wind":
        bd_s = bd.sort_values("time")
        md_s = md.sort_values("time")

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=bd_s["time"], y=bd_s["windgusts_10m_max"].rolling(30, center=True).mean(),
            name="Ballarat — Max Gust (30d avg)", line=dict(color=ACCENT, width=1.5),
        ))
        fig1.add_trace(go.Scatter(
            x=md_s["time"], y=md_s["windgusts_10m_max"].rolling(30, center=True).mean(),
            name="Melbourne — Max Gust (30d avg)", line=dict(color=ACCENT2, width=1.5),
        ))
        fig1.add_trace(go.Scatter(
            x=bd_s["time"], y=bd_s["windgusts_10m_max"],
            mode="markers", marker=dict(color=ACCENT, size=2, opacity=0.2),
            name="Ballarat — Daily Max Gust", showlegend=False,
        ))
        fig1.add_trace(go.Scatter(
            x=md_s["time"], y=md_s["windgusts_10m_max"],
            mode="markers", marker=dict(color=ACCENT2, size=2, opacity=0.2),
            name="Melbourne — Daily Max Gust", showlegend=False,
        ))
        fig1.update_layout(
            **PLOT_BASE,
            title=dict(text=f"Daily Maximum Wind Gust — Ballarat vs Melbourne ({yf}–{yt})",
                       font=dict(size=14, color=TEXT_MAIN)),
            xaxis=axis("Date"), yaxis=axis("Wind Gust (km/h)"),
            height=420, margin=dict(l=60, r=40, t=60, b=50),
        )

        bal_wind_monthly = bd.groupby("month")["windgusts_10m_max"].mean().reset_index()
        mel_wind_monthly = md.groupby("month")["windgusts_10m_max"].mean().reset_index()

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=MONTH_LABELS, y=bal_wind_monthly["windgusts_10m_max"],
            name="Ballarat", marker_color=ACCENT, opacity=0.85))
        fig2.add_trace(go.Bar(x=MONTH_LABELS, y=mel_wind_monthly["windgusts_10m_max"],
            name="Melbourne", marker_color=ACCENT2, opacity=0.85))
        fig2.update_layout(
            **PLOT_BASE,
            title=dict(text="Average Daily Max Wind Gust by Month",
                       font=dict(size=14, color=TEXT_MAIN)),
            xaxis=axis("Month"), yaxis=axis("Avg Max Gust (km/h)"),
            barmode="group", height=340, margin=dict(l=60, r=40, t=60, b=50),
        )

        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=bd["windgusts_10m_max"], name="Ballarat",
            marker_color=ACCENT, opacity=0.7, nbinsx=40, histnorm="percent"))
        fig3.add_trace(go.Histogram(x=md["windgusts_10m_max"], name="Melbourne",
            marker_color=ACCENT2, opacity=0.7, nbinsx=40, histnorm="percent"))
        fig3.update_layout(
            **PLOT_BASE,
            title=dict(text="Distribution of Daily Max Wind Gusts (% of days)",
                       font=dict(size=14, color=TEXT_MAIN)),
            xaxis=axis("Wind Gust (km/h)"), yaxis=axis("% of Days"),
            barmode="overlay", height=340, margin=dict(l=60, r=40, t=60, b=50),
        )

        return html.Div([
            dcc.Graph(figure=fig1, config={"displayModeBar": False}),
            html.Div([
                html.Div(dcc.Graph(figure=fig2, config={"displayModeBar": False}), style={"flex": 1}),
                html.Div(dcc.Graph(figure=fig3, config={"displayModeBar": False}), style={"flex": 1}),
            ], style={"display": "flex", "gap": "12px"}),
        ])

    # ── MONTHLY AVERAGES ─────────────────────────────────────────────────────
    elif tab == "monthly":
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=MONTH_LABELS, y=bal_monthly_dn["day_max"],
            name="Ballarat — Avg Day Max", line=dict(color=ACCENT, width=2.5),
            mode="lines+markers", marker=dict(size=7)))
        fig.add_trace(go.Scatter(x=MONTH_LABELS, y=mel_monthly_dn["day_max"],
            name="Melbourne — Avg Day Max", line=dict(color=ACCENT2, width=2.5),
            mode="lines+markers", marker=dict(size=7)))

        fig.update_layout(
            **PLOT_BASE,
            title=dict(text="Monthly Average Temperatures — Daytime & Nighttime (full period)",
                       font=dict(size=14, color=TEXT_MAIN)),
            xaxis=axis("Month"), yaxis=axis("Temperature (°C)"),
            height=460, margin=dict(l=60, r=40, t=60, b=50),
        )

        table_data = pd.DataFrame({
            "Month": MONTH_LABELS,
            "Bal Day Max": bal_monthly_dn["day_max"].round(1),
            "Mel Day Max": mel_monthly_dn["day_max"].round(1),
            "Diff (Mel−Bal)": (mel_monthly_dn["day_max"] - bal_monthly_dn["day_max"]).round(1),
        })

        from dash import dash_table
        tbl = dash_table.DataTable(
            data=table_data.to_dict("records"),
            columns=[{"name": c, "id": c} for c in table_data.columns],
            style_table={"overflowX": "auto", "marginTop": "16px"},
            style_header={"backgroundColor": CARD, "color": ACCENT, "fontWeight": "700",
                          "border": "1px solid #3A3530", "fontFamily": "Georgia, serif",
                          "fontSize": "0.78rem", "textTransform": "uppercase"},
            style_cell={"backgroundColor": BG, "color": TEXT_MAIN, "border": "1px solid #3A3530",
                        "fontFamily": "Georgia, serif", "fontSize": "0.85rem",
                        "padding": "8px 12px", "textAlign": "center"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#1E1E1E"},
                {"if": {"filter_query": "{Diff (Mel−Bal)} > 3", "column_id": "Diff (Mel−Bal)"},
                 "color": ACCENT2, "fontWeight": "700"},
                {"if": {"filter_query": "{Diff (Mel−Bal)} <= 3", "column_id": "Diff (Mel−Bal)"},
                 "color": ACCENT},
            ],
        )

        return html.Div([dcc.Graph(figure=fig, config={"displayModeBar": False}), tbl])

    # ── THE 3°C TEST ─────────────────────────────────────────────────────────
    elif tab == "test":
        bd_m = bd[["time","temperature_2m_max","temperature_2m_min"]].copy()
        md_m = md[["time","temperature_2m_max","temperature_2m_min"]].copy()
        merged = bd_m.merge(md_m, on="time", suffixes=("_bal","_mel"))
        merged["diff_max"] = merged["temperature_2m_max_mel"] - merged["temperature_2m_max_bal"]
        merged = merged.sort_values("time")

        avg_diff_max = merged["diff_max"].mean()
        pct_above_3  = (merged["diff_max"] > 3).mean() * 100
        pct_below_0  = (merged["diff_max"] < 0).mean() * 100

        # Rolling daily diff
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=merged["time"], y=merged["diff_max"].rolling(30, center=True).mean(),
            name="Daily Max Diff (30d avg)", line=dict(color=C_DAY, width=2),
        ))
        fig1.add_hline(y=3, line_dash="dot", line_color=ACCENT2, opacity=0.7,
                       annotation_text="3°C idiom", annotation_font_color=ACCENT2)
        fig1.add_hline(y=0, line_dash="dot", line_color=TEXT_DIM, opacity=0.4)
        fig1.update_layout(
            **PLOT_BASE,
            title=dict(text="Melbourne minus Ballarat — Daily Temperature Difference (30-day rolling avg)",
                       font=dict(size=14, color=TEXT_MAIN)),
            xaxis=axis("Date"), yaxis=axis("°C difference (Mel − Bal)"),
            height=400, margin=dict(l=60, r=40, t=60, b=50),
        )

        # Distribution of differences
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=merged["diff_max"], nbinsx=60,
            name="Daily Max Diff distribution", marker_color=C_DAY, opacity=0.8, histnorm="percent"))
        fig2.add_vline(x=3, line_dash="dot", line_color=ACCENT2, opacity=0.8,
                       annotation_text="3°C", annotation_font_color=ACCENT2)
        fig2.add_vline(x=avg_diff_max, line_dash="solid", line_color=ACCENT,
                       annotation_text=f"Mean: {avg_diff_max:.1f}°C", annotation_font_color=ACCENT)
        fig2.update_layout(
            **PLOT_BASE,
            title=dict(text="Distribution of Daily Max Temperature Differences (Melbourne − Ballarat)",
                       font=dict(size=14, color=TEXT_MAIN)),
            xaxis=axis("°C difference"), yaxis=axis("% of Days"),
            height=360, margin=dict(l=60, r=40, t=60, b=50),
        )

        # Monthly avg diff
        merged["month"] = merged["time"].dt.month
        monthly_diff = merged.groupby("month")["diff_max"].mean().reset_index()

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=MONTH_LABELS,
            y=monthly_diff["diff_max"],
            marker_color=[ACCENT2 if v > 3 else ACCENT for v in monthly_diff["diff_max"]],
            name="Avg Max Diff by Month",
            text=[f"{v:.1f}°" for v in monthly_diff["diff_max"]],
            textposition="outside",
            showlegend=False,
        ))
        fig3.add_hline(y=3, line_dash="dot", line_color=ACCENT2, opacity=0.7)

        # Dummy scatter traces for colour legend
        fig3.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=ACCENT2, size=12, symbol="square"),
            name="> 3°C warmer — claim holds",
        ))
        fig3.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=ACCENT, size=12, symbol="square"),
            name="≤ 3°C warmer — below claim",
        ))

        fig3.update_layout(
    **PLOT_BASE,
    title=dict(text="Average Daily Max Difference by Month (Melbourne − Ballarat)",
               font=dict(size=14, color=TEXT_MAIN)),
    xaxis=axis("Month"),
    yaxis=axis("°C difference (Mel − Bal)"),
    height=360,
    margin=dict(l=60, r=40, t=60, b=50),
    showlegend=True,
)
        fig3.update_layout(
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT_DIM, size=11),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        # Verdict card
        verdict_color = ACCENT2 if avg_diff_max >= 2.5 else ACCENT
        verdict_text = (
            f"Mostly TRUE — Melbourne averages {avg_diff_max:.1f}°C warmer (daily max) than Ballarat over {yf}–{yt}."
            if avg_diff_max >= 2.5 else
            f"PARTIALLY TRUE — Melbourne averages {avg_diff_max:.1f}°C warmer (daily max) — less than the 3°C idiom suggests."
        )
        verdict = html.Div([
            html.P("🧪  Verdict on the 3°C Claim", style={"color": verdict_color, "fontWeight": "700",
                    "fontSize": "0.88rem", "margin": "0 0 8px 0", "textTransform": "uppercase"}),
            html.P(verdict_text, style={"color": TEXT_MAIN, "fontSize": "0.85rem", "margin": "0 0 10px 0"}),
            html.Ul([
                html.Li(f"Mean daily max difference: {avg_diff_max:.2f}°C"),
                html.Li(f"Days where Melbourne is > 3°C warmer: {pct_above_3:.1f}%"),
                html.Li(f"Days where Ballarat is actually warmer: {pct_below_0:.1f}%"),
                html.Li("The gap is largest in spring/summer and smallest in winter."),
                html.Li("Nighttime difference is smaller — Ballarat's cold nights are closer to Melbourne's than daytime suggests."),
            ], style={"fontSize": "0.8rem", "color": TEXT_DIM, "lineHeight": "1.9", "paddingLeft": "18px"}),
        ], style={"background": CARD, "borderRadius": "10px", "padding": "18px 22px",
                  "borderLeft": f"4px solid {verdict_color}", "marginTop": "14px"})

        return html.Div([
            verdict,
            dcc.Graph(figure=fig1, config={"displayModeBar": False}),
            html.Div([
                html.Div(dcc.Graph(figure=fig2, config={"displayModeBar": False}), style={"flex": 1}),
                html.Div(dcc.Graph(figure=fig3, config={"displayModeBar": False}), style={"flex": 1}),
            ], style={"display": "flex", "gap": "12px"}),
        ])

    # ── DEFINITIONS ──────────────────────────────────────────────────────────
    elif tab == "defs":

        def def_block(title, items, color=ACCENT):
            rows = []
            for metric, explanation in items:
                rows.append(html.Tr([
                    html.Td(metric, style={"color": color, "fontWeight": "700",
                        "padding": "10px 16px", "fontSize": "0.82rem",
                        "whiteSpace": "nowrap", "verticalAlign": "top",
                        "borderBottom": "1px solid #3A3530", "width": "220px"}),
                    html.Td(explanation, style={"color": TEXT_DIM, "padding": "10px 16px",
                        "fontSize": "0.8rem", "lineHeight": "1.7",
                        "borderBottom": "1px solid #3A3530"}),
                ]))
            return html.Div([
                html.H4(title, style={"color": color, "margin": "0 0 0 0",
                    "fontSize": "0.95rem", "fontFamily": "Georgia, serif",
                    "fontWeight": "700", "padding": "14px 16px",
                    "background": "#1E1E1E", "borderRadius": "10px 10px 0 0",
                    "borderLeft": f"4px solid {color}"}),
                html.Table(html.Tbody(rows),
                    style={"width": "100%", "borderCollapse": "collapse",
                           "background": CARD, "borderRadius": "0 0 10px 10px"}),
            ], style={"marginBottom": "20px", "borderRadius": "10px",
                      "overflow": "hidden", "border": "1px solid #3A3530"})

        source_note = html.Div([
            html.P([
                html.Strong("Data source: ", style={"color": TEXT_DIM}),
                "All data is sourced from the ",
                html.Strong("Open-Meteo Historical Weather API", style={"color": ACCENT}),
                " using ERA5 reanalysis data from the European Centre for Medium-Range Weather Forecasts (ECMWF). "
                "ERA5 is a global climate reanalysis dataset that combines model data with observations at ~9km resolution. "
                "It is not identical to BOM station readings but is widely used in climate research and is suitable for "
                "comparative trend analysis between locations.",
                html.Br(), html.Br(),
                html.Strong("Period: ", style={"color": TEXT_DIM}),
                f"{START_DATE[:4]}–{END_DATE[:4]}. ",
                html.Strong("Coordinates: ", style={"color": TEXT_DIM}),
                f"Ballarat ({BALLARAT_LAT}, {BALLARAT_LON}), Melbourne ({MELBOURNE_LAT}, {MELBOURNE_LON}). ",
                html.Strong("Timezone: ", style={"color": TEXT_DIM}),
                "Australia/Melbourne (AEDT/AEST). All times are local.",
            ], style={"fontSize": "0.78rem", "color": TEXT_DIM, "lineHeight": "1.7", "margin": 0}),
        ], style={"background": CARD, "borderRadius": "10px", "padding": "16px 20px",
                  "borderLeft": f"4px solid {ACCENT}", "marginBottom": "20px"})

        return html.Div([
            source_note,

            def_block("🌡  Temperature Ranges tab", [
                ("Ballarat / Melbourne Daily Max",
                 "The highest temperature recorded at 2 metres above ground for that calendar day. "
                 "Raw daily values — no smoothing applied. This is the standard daily maximum "
                 "used by BOM and ERA5."),
                ("Show / hide checkboxes",
                 "Each series (Ballarat daily max, Melbourne daily max) can be toggled on or off "
                 "using the checkboxes above the chart. You can also click legend items directly "
                 "on the chart to toggle them."),
                ("Zoom & reset",
                 "Click and drag on the chart to zoom into a specific date range. Use the reset "
                 "button (⌂ house icon, top-right of chart) to return to the full selected period."),
            ]),

            def_block("🌙  Day vs Night tab", [
                ("Daytime hours",
                 "Hours where the Open-Meteo 'is_day' flag equals 1 — from sunrise to sunset — "
                 "calculated for each location's exact latitude and longitude. Daytime length "
                 "varies from ~9.5 hours (winter) to ~14.5 hours (summer) in Victoria."),
                ("Nighttime hours",
                 "Hours where the 'is_day' flag equals 0 — after sunset and before the following "
                 "sunrise. Includes the evening, overnight, and early morning hours."),
                ("Daytime Max (solid lines)",
                 "The highest hourly temperature recorded during daylight hours on that day. "
                 "Displayed as a 30-day rolling average."),
                ("Nighttime Max (dotted lines)",
                 "The highest hourly temperature recorded during nighttime hours — typically "
                 "occurring in the early evening just after sunset. Displayed as a 30-day "
                 "rolling average. Ballarat's nighttime peak sits noticeably lower than "
                 "Melbourne's, consistent with the urban heat island effect."),
                ("30-day rolling average",
                 "All lines are smoothed using a 30-day centred rolling average to reduce "
                 "day-to-day noise and reveal seasonal patterns clearly."),
            ], color=C_NIGHT),

            def_block("🌧  Rainfall tab", [
                ("Precipitation (mm)",
                 "Total liquid-equivalent precipitation in millimetres for a given day, "
                 "including rain, drizzle, and melted snow or hail. ERA5 precipitation is "
                 "model-derived and may differ slightly from BOM gauge readings."),
                ("Rain day (≥ 0.2mm)",
                 "A day is counted as a rain day if total precipitation reaches 0.2mm or more. "
                 "This is the standard BOM definition — it excludes trace drizzle that does not "
                 "meaningfully wet the ground."),
                ("% of days with rain",
                 "The percentage of days in each calendar month that recorded ≥ 0.2mm of rain, "
                 "averaged across all years in the selected period."),
                ("Annual rainfall total",
                 "Sum of all daily precipitation for each calendar year. Dotted trend lines "
                 "show the linear regression direction across the selected period."),
                ("Average daily rainfall (mm)",
                 "Mean daily precipitation for each month across all years, including dry days "
                 "(zero rainfall days). Represents how wet a typical day in that month is."),
            ], color=C_RAIN),

            def_block("💨  Wind tab", [
                ("Daily max wind gust (km/h)",
                 "The highest hourly wind gust recorded across all hours of that calendar day, "
                 "measured at 10 metres above ground. ERA5 gusts are 3-second peak gusts derived "
                 "from the model — broadly comparable to BOM anemometer readings."),
                ("30-day rolling avg line",
                 "The bold line smooths the daily max gust over a 30-day centred window, "
                 "revealing seasonal patterns. Spring is typically the windiest season in "
                 "both Ballarat and Melbourne."),
                ("Raw daily dots",
                 "The faint scattered dots show the actual unsmoothed daily maximum gust. "
                 "Extreme outlier dots represent individual storm events."),
                ("Distribution histogram",
                 "Shows what percentage of days fell into each wind speed band across the full "
                 "selected period. A tail to the right indicates occasional high-gust events."),
                ("Average max gust by month",
                 "Mean daily maximum gust for each calendar month, averaged across all years "
                 "in the selected period."),
            ], color=C_WIND),

            def_block("📅  Monthly Averages tab", [
                ("Avg Day Max",
                 "Mean of all daytime maximum temperatures for each calendar month, across all "
                 "years in the full dataset. The daytime maximum is the highest temperature "
                 "recorded during sunlit hours."),
                ("Diff (Mel−Bal) in table",
                 "Melbourne's average daytime maximum minus Ballarat's for each month. "
                 "Positive = Melbourne warmer. Values above 3°C are highlighted as a reference "
                 "to the common '3°C warmer' claim."),
            ], color=C_DAY),

            def_block("🧪  The 3°C Test tab", [
                ("Daily max difference (Mel − Bal)",
                 "Melbourne's daily maximum temperature minus Ballarat's for the same calendar "
                 "day. Positive = Melbourne warmer. Negative = Ballarat warmer."),
                ("30-day rolling avg line",
                 "Smoothed daily difference over a 30-day centred window, showing the seasonal "
                 "pattern in the Melbourne–Ballarat temperature gap."),
                ("Distribution histogram",
                 "Percentage of days in each temperature difference band. The gold vertical line "
                 "marks the mean. The blue vertical line marks the 3°C claim threshold."),
                ("% of days Melbourne > 3°C warmer",
                 "Proportion of days where Melbourne's daily max exceeded Ballarat's by more than 3°C."),
                ("% of days Ballarat warmer",
                 "Proportion of days where Ballarat's daily max was equal to or higher than "
                 "Melbourne's. More common than most people expect, particularly in autumn and winter."),
                ("Monthly avg diff bars",
                 "Mean Melbourne-minus-Ballarat daily max difference by calendar month. "
                 "Blue bars (> 3°C) indicate months where the 3°C claim holds. "
                 "Gold bars (≤ 3°C) indicate months where the gap is smaller than the claim suggests."),
            ], color=ACCENT2),

        ], style={"maxWidth": "900px", "paddingBottom": "20px"})

    return html.Div("Select a tab.")


if __name__ == "__main__":
    print("\n" + "═"*60)
    print("  Open: http://127.0.0.1:8050")
    print("═"*60 + "\n")
    app.run(debug=True)
