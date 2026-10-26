"""
ui.py
=====
Presentation layer for the Student Performance Prediction app.

Everything visual lives here so `app.py` can stay focused on behaviour:

* ``CSS``      - one stylesheet for the whole app (theme, cards, tabs, charts)
* ``inject()`` - writes that stylesheet into the page
* builders     - small functions that return single-line HTML fragments
                 (hero, stat tiles, score ring, probability bars, tables,
                  confusion matrix, advice cards, empty state)

Design notes
------------
* Indigo / violet accent on a cool off-white background, Inter typeface.
* Every "card" is either real HTML (static content) or a styled Streamlit
  element (forms, charts, dataframes) so widgets keep working normally.
* All HTML is emitted as one line: Streamlit renders markdown, and a line
  indented by four spaces would become a code block instead.
"""

from __future__ import annotations

import html
import math
import re

import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
INK = "#0b1220"
BODY = "#475569"
MUTED = "#7b869c"
LINE = "#e6eaf3"
BRAND = "#4f46e5"
BRAND_2 = "#7c3aed"
CYAN = "#06b6d4"
GREEN = "#15803d"
AMBER = "#b45309"
RED = "#b91c1c"

TONES = {
  "brand": {"color": BRAND, "soft": "#eef2ff"},
  "violet": {"color": "#7c3aed", "soft": "#f5f3ff"},
  "green": {"color": GREEN, "soft": "#dcfce7"},
  "amber": {"color": AMBER, "soft": "#fef3c7"},
  "red": {"color": RED, "soft": "#fee2e2"},
  "cyan": {"color": "#0e7490", "soft": "#cffafe"},
  "slate": {"color": "#475569", "soft": "#f1f5f9"},
}

PLOT_FONT = {
  "family": "Inter, Segoe UI, system-ui, -apple-system, sans-serif",
  "size": 12,
  "color": "#475569",
}
GRID = "#eef2f8"


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --ink: #0b1220;
  --body: #475569;
  --muted: #7b869c;
  --line: #e6eaf3;
  --brand: #4f46e5;
  --brand-2: #7c3aed;
  --surface: #ffffff;
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
  font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.stApp {
  background:
    radial-gradient(1100px 520px at 10% -12%, #e7ecff 0%, rgba(231,236,255,0) 58%),
    radial-gradient(900px 460px at 98% -6%, #e4f6ff 0%, rgba(228,246,255,0) 55%),
    #f4f6fb;
  color: var(--ink);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
#MainMenu, footer, [data-testid="stHeaderActionElements"] { display: none !important; }
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 1240px;
  padding-top: 1.6rem;
  padding-bottom: 3.5rem;
}
[data-testid="stVerticalBlock"] { gap: 0.85rem; }

h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.02em; }
p, li { color: var(--body); }
a { color: var(--brand); }
::selection { background: #ddd6fe; color: #1e1b4b; }

/* Scrollbar ------------------------------------------------------------- */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: #cfd7e8; border-radius: 999px; border: 3px solid #f4f6fb; }
::-webkit-scrollbar-track { background: transparent; }

/* Hero ------------------------------------------------------------------ */
.hero {
  position: relative;
  overflow: hidden;
  border-radius: 26px;
  padding: 34px 38px 30px;
  margin-bottom: 8px;
  background:
    radial-gradient(760px 320px at 4% -24%, rgba(129,140,248,.55), transparent 62%),
    radial-gradient(640px 300px at 94% -10%, rgba(34,211,238,.30), transparent 60%),
    linear-gradient(125deg, #080d1f 0%, #121a3c 46%, #1e2657 100%);
  color: #f8fafc;
  box-shadow: 0 36px 64px -34px rgba(11,18,32,.72);
}
.hero::before {
  content: "";
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,.055) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.055) 1px, transparent 1px);
  background-size: 46px 46px;
  -webkit-mask-image: radial-gradient(85% 110% at 72% 0%, #000 18%, transparent 76%);
  mask-image: radial-gradient(85% 110% at 72% 0%, #000 18%, transparent 76%);
}
.hero::after {
  content: "";
  position: absolute;
  width: 420px;
  height: 420px;
  right: -110px;
  top: -170px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(124,58,237,.55), transparent 66%);
  filter: blur(4px);
}
.hero > * { position: relative; z-index: 1; }
.hero-mark {
  position: absolute;
  right: 18px;
  bottom: -46px;
  width: 300px;
  height: 300px;
  opacity: .30;
  z-index: 0;
  pointer-events: none;
}
.hero-mark svg { width: 100%; height: 100%; }
.hero .kicker {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: .67rem;
  letter-spacing: .17em;
  text-transform: uppercase;
  font-weight: 700;
  color: #c7d2fe;
  background: rgba(255,255,255,.07);
  border: 1px solid rgba(199,210,254,.24);
  padding: 6px 12px;
  border-radius: 999px;
}
.hero .kicker .pulse {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #34d399;
  box-shadow: 0 0 0 4px rgba(52,211,153,.18);
}
.hero h1 {
  font-size: 2.3rem;
  line-height: 1.12;
  margin: 15px 0 10px;
  font-weight: 800;
  letter-spacing: -.035em;
  max-width: 820px;
  color: #f8fafc;
}
.hero h1 .grad {
  background: linear-gradient(100deg, #a5b4fc 0%, #67e8f9 55%, #c4b5fd 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.hero p {
  margin: 0;
  max-width: 700px;
  color: #b8c3da;
  font-size: .93rem;
  line-height: 1.65;
}
.hero .facts { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 22px; }
.hero .fact {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  font-size: .74rem;
  color: #cbd5e1;
  background: rgba(255,255,255,.07);
  border: 1px solid rgba(255,255,255,.12);
  padding: 7px 12px;
  border-radius: 11px;
}
.hero .fact b { color: #ffffff; font-weight: 700; }

/* Section headers ------------------------------------------------------- */
.section {
  display: flex;
  align-items: center;
  gap: 11px;
  margin: 30px 0 14px;
}
.section-icon {
  width: 34px;
  height: 34px;
  border-radius: 11px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #eef2ff, #e0f2fe);
  border: 1px solid #e0e7ff;
  font-size: 1rem;
  flex: 0 0 auto;
}
.section h2 {
  font-size: 1.06rem;
  margin: 0;
  font-weight: 700;
  letter-spacing: -.015em;
  white-space: nowrap;
}
.section-line {
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, #dde4f1, rgba(221,228,241,0));
}
.section-note {
  font-size: .72rem;
  color: var(--muted);
  font-weight: 500;
  white-space: nowrap;
}

/* Tabs ------------------------------------------------------------------ */
[data-testid="stTabs"] { margin-top: 10px; }
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 5px;
  width: max-content !important;
  background: #e9edf7;
  padding: 5px;
  border-radius: 999px;
  border: 1px solid #e0e6f2;
  box-shadow: inset 0 1px 2px rgba(15,23,42,.04);
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  height: auto;
  padding: 8px 18px !important;
  border: none !important;
  border-radius: 999px !important;
  background: transparent !important;
  color: #5b6478;
  font-size: .87rem;
  font-weight: 600;
  transition: all .16s ease;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
  background: rgba(255,255,255,.72) !important;
  color: #1e293b;
}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
  background: #ffffff !important;
  color: #3730a3 !important;
  box-shadow: 0 5px 14px -6px rgba(15,23,42,.28);
}
[data-testid="stTabs"] [data-baseweb="tab"] p { font-size: .87rem; font-weight: 600; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }
[data-baseweb="tab-panel"] { padding-top: 16px; }

/* Forms ----------------------------------------------------------------- */
[data-testid="stForm"] {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 22px;
  padding: 22px 22px 18px;
  box-shadow: 0 26px 54px -40px rgba(15,23,42,.45);
}
[data-testid="stForm"] [data-testid="stVerticalBlock"] { gap: 0.55rem; }

/* Widgets --------------------------------------------------------------- */
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label p {
  font-size: .83rem !important;
  font-weight: 600 !important;
  color: #334155 !important;
}
[data-testid="stCaptionContainer"] p {
  font-size: .73rem !important;
  color: var(--muted) !important;
  line-height: 1.5;
}
[data-testid="stTextInput"] input {
  border-radius: 11px;
  border: 1px solid #e2e8f0;
  background: #fbfcfe;
  padding: .52rem .72rem;
  font-size: .87rem;
  color: var(--ink);
}
[data-testid="stTextInput"] input:focus {
  border-color: #a5b4fc;
  box-shadow: 0 0 0 4px rgba(99,102,241,.14);
  background: #ffffff;
}
[data-testid="stSlider"] { padding: 2px 0 4px; }
[data-testid="stSlider"] [data-testid="stThumbValue"] {
  color: #4338ca;
  font-weight: 700;
  font-size: .78rem;
}
[data-testid="stSlider"] [data-testid="stTickBar"],
[data-testid="stSlider"] [data-testid="stSliderTickBar"],
[data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
[data-testid="stSlider"] [data-testid="stSliderTickBarMax"] { display: none !important; }
[data-testid="stSlider"] div[role="slider"] {
  background: var(--brand);
  box-shadow: 0 0 0 4px rgba(79,70,229,.16);
}
.stButton > button, .stDownloadButton > button {
  border-radius: 11px;
  border: 1px solid var(--line);
  background: #ffffff;
  color: #28324a;
  font-size: .83rem;
  font-weight: 600;
  transition: all .16s ease;
  box-shadow: 0 1px 2px rgba(15,23,42,.04);
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: #c7d2fe;
  color: #3730a3;
  transform: translateY(-1px);
  box-shadow: 0 10px 22px -12px rgba(79,70,229,.6);
}
[data-testid="stFormSubmitButton"] button {
  background: linear-gradient(118deg, #4f46e5 0%, #6d28d9 55%, #7c3aed 100%);
  border: none;
  color: #ffffff;
  font-size: .94rem;
  font-weight: 700;
  padding: .72rem 1rem;
  box-shadow: 0 16px 30px -14px rgba(79,70,229,.85);
}
[data-testid="stFormSubmitButton"] button:hover {
  color: #ffffff;
  transform: translateY(-1px);
  box-shadow: 0 22px 38px -14px rgba(79,70,229,.9);
}
[data-testid="stFormSubmitButton"] button p { font-weight: 700; }

/* Card-like Streamlit elements ----------------------------------------- */
[data-testid="stForm"], [data-testid="stPlotlyChart"], [data-testid="stDataFrame"],
[data-testid="stExpander"], [data-testid="stImage"] img {
  background: #ffffff;
}
[data-testid="stPlotlyChart"] {
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 12px 12px 4px;
  box-shadow: 0 20px 44px -38px rgba(15,23,42,.55);
  overflow: hidden;
}
[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 16px 36px -36px rgba(15,23,42,.45);
}
[data-testid="stExpander"] {
  border: 1px solid var(--line) !important;
  border-radius: 16px;
  box-shadow: none;
  overflow: hidden;
}
[data-testid="stExpander"] summary {
  font-size: .86rem;
  font-weight: 600;
  color: #334155;
  padding: .7rem .95rem;
}
[data-testid="stExpander"] summary:hover { color: #4338ca; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] { padding: 0 .95rem .6rem; }
[data-testid="stImage"] img {
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 8px;
}
[data-testid="stAlert"] {
  border-radius: 14px;
  border: 1px solid #fde68a;
  background: #fffbeb;
}
[data-testid="stCode"] pre { border-radius: 12px; }

/* Generic panels -------------------------------------------------------- */
.panel {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 20px 22px;
  box-shadow: 0 22px 48px -40px rgba(15,23,42,.55);
}
.panel-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 15px;
}
.panel-title { font-size: .94rem; font-weight: 700; color: var(--ink); letter-spacing: -.01em; }
.panel-sub { font-size: .74rem; color: var(--muted); margin-top: 3px; line-height: 1.45; }
.panel-note {
  font-size: .64rem;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: #94a3b8;
  white-space: nowrap;
  padding-top: 3px;
}

/* Stat tiles ------------------------------------------------------------ */
.tiles {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(186px, 1fr));
  gap: 14px;
}
.tile {
  position: relative;
  overflow: hidden;
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 15px 17px 14px;
  box-shadow: 0 18px 38px -34px rgba(15,23,42,.5);
}
.tile::after {
  content: "";
  position: absolute;
  right: -34px;
  bottom: -52px;
  width: 118px;
  height: 118px;
  border-radius: 50%;
  background: var(--tone-soft, #eef2ff);
  opacity: .75;
}
.tile-top { display: flex; align-items: center; gap: 9px; }
.tile-ico {
  width: 30px;
  height: 30px;
  border-radius: 9px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: .9rem;
  background: var(--tone-soft, #eef2ff);
  border: 1px solid rgba(15,23,42,.04);
  flex: 0 0 auto;
}
.tile-label {
  font-size: .63rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: #8a94a8;
  font-weight: 700;
}
.tile-value {
  position: relative;
  z-index: 1;
  font-size: 1.6rem;
  font-weight: 800;
  letter-spacing: -.035em;
  color: var(--tone, var(--ink));
  line-height: 1.1;
  margin-top: 9px;
}
.tile-hint {
  position: relative;
  z-index: 1;
  font-size: .72rem;
  color: var(--muted);
  line-height: 1.45;
  margin-top: 3px;
}

/* Result card ----------------------------------------------------------- */
.result-card {
  position: relative;
  overflow: hidden;
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 24px;
  padding: 24px 24px 20px;
  box-shadow: 0 30px 60px -42px rgba(15,23,42,.6);
}
.result-card::before {
  content: "";
  position: absolute;
  left: 0;
  right: 0;
  top: 0;
  height: 5px;
  background: linear-gradient(90deg, var(--band), var(--band-2));
}
.result-grid {
  display: flex;
  align-items: center;
  gap: 20px;
  justify-content: space-between;
  flex-wrap: wrap;
}
.result-copy { flex: 1 1 270px; min-width: 240px; }
.result-eyebrow {
  font-size: .64rem;
  letter-spacing: .16em;
  text-transform: uppercase;
  font-weight: 800;
  color: #8a94a8;
}
.result-band {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 2.15rem;
  font-weight: 800;
  letter-spacing: -.04em;
  line-height: 1.05;
  margin: 8px 0 8px;
  color: var(--band);
}
.result-band .emoji { font-size: 1.7rem; }
.result-sub {
  margin: 0 0 14px;
  font-size: .86rem;
  color: #5b6478;
  line-height: 1.6;
  max-width: 430px;
}
.result-conf {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: .76rem;
  color: #7b869c;
  margin: -1px 0 11px;
}
.result-conf b { font-weight: 700; color: #334155; }
.result-conf .sep { color: #cbd5e1; }
.result-who {
  margin-top: 13px;
  padding-top: 11px;
  border-top: 1px dashed var(--line);
  font-size: .75rem;
  color: var(--muted);
}
.result-who b { color: #334155; font-weight: 600; }

/* Chips ----------------------------------------------------------------- */
.chips { display: flex; flex-wrap: wrap; gap: 7px; }
.chip {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  font-size: .73rem;
  font-weight: 500;
  padding: 6px 10px;
  border-radius: 999px;
  background: #f7f9fd;
  border: 1px solid #e6eaf3;
  color: #4b5568;
  white-space: nowrap;
}
.chip b { font-weight: 700; }
.dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; flex: 0 0 auto; }

/* Score ring ------------------------------------------------------------ */
.ring-wrap {
  position: relative;
  width: 184px;
  height: 184px;
  flex: 0 0 auto;
  margin: 0 auto;
}
.ring-wrap svg { width: 184px; height: 184px; transform: rotate(-90deg); }
.ring-track { fill: none; stroke: #eef2f9; stroke-width: 13; }
.ring-fill { fill: none; stroke-width: 13; stroke-linecap: round; }
.ring-center {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
}
.ring-value {
  font-size: 2rem;
  font-weight: 800;
  letter-spacing: -.045em;
  color: var(--ink);
  line-height: 1;
}
.ring-unit {
  font-size: .58rem;
  font-weight: 800;
  letter-spacing: .16em;
  text-transform: uppercase;
  color: #9aa4b8;
  margin-top: 5px;
}
.ring-label {
  font-size: .66rem;
  font-weight: 600;
  color: #64748b;
  margin-top: 8px;
  padding: 3px 9px;
  border-radius: 999px;
  background: var(--band-soft, #eef2ff);
  color: var(--band, #4f46e5);
}

/* Probability bars ------------------------------------------------------ */
.prob-list { display: flex; flex-direction: column; gap: 8px; }
.prob-row {
  display: grid;
  grid-template-columns: 96px 1fr 54px;
  align-items: center;
  gap: 11px;
  padding: 5px 8px;
  margin: 0 -8px;
  border-radius: 11px;
}
.prob-row.is-predicted { background: #f7f9ff; }
.prob-label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: .78rem;
  font-weight: 600;
  color: #64748b;
}
.prob-row.is-predicted .prob-label { color: var(--ink); }
.prob-track {
  height: 9px;
  border-radius: 999px;
  background: #eff2f8;
  overflow: hidden;
}
.prob-fill {
  height: 100%;
  border-radius: 999px;
  background: var(--bar, #4f46e5);
  opacity: .55;
}
.prob-row.is-predicted .prob-fill { opacity: 1; }
.prob-val {
  font-size: .78rem;
  font-weight: 700;
  color: #94a3b8;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.prob-row.is-predicted .prob-val { color: var(--band); }

/* Advice / tip cards ---------------------------------------------------- */
.tip {
  border: 1px solid var(--line);
  border-left: 4px solid var(--tone);
  border-radius: 16px;
  background: #ffffff;
  padding: 15px 17px;
  margin-bottom: 12px;
  box-shadow: 0 16px 34px -34px rgba(15,23,42,.55);
  transition: transform .16s ease, box-shadow .16s ease;
}
.tip:hover {
  transform: translateY(-2px);
  box-shadow: 0 24px 42px -32px rgba(15,23,42,.5);
}
.tip-head { display: flex; align-items: center; gap: 10px; margin-bottom: 9px; }
.tip-icon {
  width: 34px;
  height: 34px;
  border-radius: 10px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: .95rem;
  background: var(--tone-soft);
  border: 1px solid rgba(15,23,42,.05);
  flex: 0 0 auto;
}
.tip-area { font-size: .88rem; font-weight: 700; color: var(--ink); line-height: 1.25; }
.tip-sub { font-size: .66rem; color: var(--muted); margin-top: 2px; }
.badge {
  margin-left: auto;
  font-size: .58rem;
  font-weight: 800;
  letter-spacing: .09em;
  text-transform: uppercase;
  color: #ffffff;
  background: var(--tone);
  padding: 3px 8px;
  border-radius: 999px;
  white-space: nowrap;
}
.tip-msg {
  margin: 0;
  font-size: .82rem;
  line-height: 1.62;
  color: #4b5568;
}
.tip-action {
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px dashed var(--line);
  font-size: .81rem;
  line-height: 1.6;
  color: #334155;
}
.tip-action span {
  display: block;
  font-size: .59rem;
  letter-spacing: .12em;
  text-transform: uppercase;
  font-weight: 800;
  color: var(--tone);
  margin-bottom: 3px;
}

/* Callout --------------------------------------------------------------- */
.callout {
  display: flex;
  align-items: flex-start;
  gap: 13px;
  border-radius: 18px;
  padding: 15px 17px;
  border: 1px solid var(--tone-soft);
  background: linear-gradient(115deg, #ffffff, var(--tone-soft));
  margin-bottom: 14px;
}
.callout .ico {
  width: 34px;
  height: 34px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #ffffff;
  font-size: 1rem;
  box-shadow: 0 8px 18px -12px rgba(15,23,42,.5);
  flex: 0 0 auto;
}
.callout .ttl { font-size: .87rem; font-weight: 700; color: var(--tone); line-height: 1.35; }
.callout .sub { font-size: .78rem; color: #4b5568; line-height: 1.55; margin-top: 3px; }
.callout.warn {
  border-color: #fde68a;
  background: linear-gradient(115deg, #ffffff, #fffbeb);
}
.callout.warn .ttl { color: #92400e; }

/* Soft note ------------------------------------------------------------- */
.softnote {
  display: flex;
  gap: 9px;
  align-items: flex-start;
  font-size: .76rem;
  line-height: 1.55;
  color: #64748b;
  background: #f7f9fd;
  border: 1px solid #eaeef7;
  border-radius: 13px;
  padding: 10px 12px;
  margin: 4px 0 6px;
}
.softnote .i { flex: 0 0 auto; }

/* Empty state ----------------------------------------------------------- */
.empty {
  background: #ffffff;
  border: 1px dashed #d3dcee;
  border-radius: 24px;
  padding: 40px 26px;
  text-align: center;
  box-shadow: inset 0 0 0 1px #ffffff;
}
.empty .art {
  width: 92px;
  height: 92px;
  margin: 0 auto 18px;
  border-radius: 26px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2.1rem;
  background:
    radial-gradient(circle at 28% 18%, #ffffff 0%, #eef2ff 45%, #e0f2fe 100%);
  border: 1px solid #e0e7ff;
  box-shadow: 0 22px 42px -26px rgba(79,70,229,.65);
}
.empty h3 {
  margin: 0 0 7px;
  font-size: 1.12rem;
  font-weight: 700;
  letter-spacing: -.02em;
  color: var(--ink);
}
.empty p {
  margin: 0 auto 18px;
  max-width: 350px;
  font-size: .84rem;
  line-height: 1.62;
  color: #64748b;
}
.empty .quick { display: flex; flex-wrap: wrap; gap: 7px; justify-content: center; }
.empty .quick span {
  font-size: .7rem;
  font-weight: 600;
  color: #4f46e5;
  background: #eef2ff;
  border: 1px solid #e0e7ff;
  padding: 5px 10px;
  border-radius: 999px;
}

/* Tables ---------------------------------------------------------------- */
.panel.tbl-panel { padding: 16px 0 4px; }
.tbl-panel .panel-head { padding: 0 18px; margin-bottom: 12px; }
table.tbl { width: 100%; border-collapse: collapse; font-size: .82rem; }
.tbl thead th {
  text-align: left;
  font-size: .62rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: #8a94a8;
  font-weight: 800;
  padding: 9px 18px;
  background: #f8fafc;
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  white-space: nowrap;
}
.tbl td {
  padding: 9px 18px;
  border-bottom: 1px solid #f1f4fa;
  color: #3f4a61;
  font-variant-numeric: tabular-nums;
}
.tbl tbody tr:last-child td { border-bottom: none; }
.tbl tbody tr:hover td { background: #fafbff; }
.tbl tr.hl td { background: #f5f6ff; font-weight: 700; color: #312e81; }
.tbl td.num, .tbl th.num { text-align: right; }
.tbl .bandcell { display: inline-flex; align-items: center; gap: 6px; font-weight: 600; }
.tbl td .dot { margin-right: 6px; vertical-align: 1px; }

/* Confusion matrix ------------------------------------------------------ */
.cm-grid {
  display: grid;
  grid-template-columns: 82px repeat(4, 1fr);
  gap: 6px;
  align-items: stretch;
}
.cm-head {
  font-size: .61rem;
  font-weight: 800;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: #8a94a8;
  text-align: center;
  padding-bottom: 2px;
}
.cm-side {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 5px;
  font-size: .72rem;
  font-weight: 700;
  color: #475569;
  padding-right: 4px;
}
.cm-cell {
  min-height: 56px;
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1px;
  font-size: .98rem;
  font-weight: 800;
  font-variant-numeric: tabular-nums;
  border: 1px solid transparent;
}
.cm-cell small { font-size: .6rem; font-weight: 600; opacity: .8; }
.cm-cell.diag { border-color: rgba(79,70,229,.45); }
.cm-axes {
  display: flex;
  justify-content: space-between;
  margin-top: 11px;
  font-size: .62rem;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: #a3adc0;
}

/* Small rows list ------------------------------------------------------- */
.rowlist { display: flex; flex-direction: column; gap: 8px; }
.rowitem {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 9px 11px;
  border-radius: 12px;
  background: #f8fafc;
  border: 1px solid #eef2f8;
}
.rowitem .k { font-size: .78rem; font-weight: 600; color: #334155; }
.rowitem .v {
  margin-left: auto;
  font-size: .77rem;
  font-weight: 700;
  color: #334155;
  font-variant-numeric: tabular-nums;
}
.rowitem .s {
  font-size: .57rem;
  font-weight: 800;
  letter-spacing: .07em;
  text-transform: uppercase;
  padding: 3px 7px;
  border-radius: 999px;
  white-space: nowrap;
}

/* Weight-meter rows ----------------------------------------------------- */
.weights { display: flex; flex-direction: column; gap: 10px; }
.weight-row { display: grid; grid-template-columns: 108px 1fr 44px; gap: 10px; align-items: center; }
.weight-name { font-size: .75rem; font-weight: 600; color: #4b5568; }
.weight-track { height: 7px; border-radius: 999px; background: #eff2f8; overflow: hidden; }
.weight-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--tone), var(--tone-soft)); }
.weight-val {
  font-size: .72rem;
  font-weight: 700;
  color: #64748b;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

/* Sidebar --------------------------------------------------------------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #ffffff 0%, #fbfcff 100%);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.5rem; }
[data-testid="stSidebar"] .stButton > button {
  width: 100%;
  justify-content: flex-start;
  text-align: left;
  padding: 7px 11px;
  font-size: .79rem;
  font-weight: 600;
  border-radius: 10px;
  background: #ffffff;
}
[data-testid="stSidebar"] .stButton > button p { text-align: left; font-size: .79rem; }
[data-testid="stSidebar"] .stButton > button:hover { background: #f8faff; }
.sb-brand { display: flex; align-items: center; gap: 11px; }
.sb-logo {
  width: 42px;
  height: 42px;
  border-radius: 13px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.15rem;
  background: linear-gradient(135deg, #4f46e5, #7c3aed 55%, #06b6d4);
  box-shadow: 0 14px 26px -14px rgba(79,70,229,.85);
  flex: 0 0 auto;
}
.sb-name { font-size: .9rem; font-weight: 800; letter-spacing: -.02em; color: var(--ink); line-height: 1.15; }
.sb-sub { font-size: .67rem; color: #8a94a8; font-weight: 500; margin-top: 2px; }
.sb-label {
  font-size: .59rem;
  letter-spacing: .15em;
  text-transform: uppercase;
  font-weight: 800;
  color: #9aa4b8;
  margin: 4px 0 2px;
}
.sb-note { font-size: .7rem; color: #8a94a8; line-height: 1.55; }
.sb-card {
  background: linear-gradient(145deg, #f7f9ff, #eef2ff);
  border: 1px solid #e3e9f8;
  border-radius: 15px;
  padding: 12px 14px;
}
.sb-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
  font-size: .71rem;
  padding: 5px 0;
  border-bottom: 1px dashed #e3e9f8;
}
.sb-row:last-child { border-bottom: none; padding-bottom: 0; }
.sb-row .k { color: #7b869c; }
.sb-row .v { color: #312e81; font-weight: 700; }
.steps { list-style: none; margin: 0; padding: 0; counter-reset: s; }
.steps li {
  position: relative;
  padding-left: 27px;
  margin-bottom: 11px;
  font-size: .74rem;
  color: #5b6478;
  line-height: 1.5;
}
.steps li:last-child { margin-bottom: 0; }
.steps li::before {
  counter-increment: s;
  content: counter(s);
  position: absolute;
  left: 0;
  top: 0;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: #eef2ff;
  border: 1px solid #e0e7ff;
  color: #4f46e5;
  font-size: .6rem;
  font-weight: 800;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* Footer ---------------------------------------------------------------- */
.foot {
  margin-top: 42px;
  padding-top: 16px;
  border-top: 1px solid var(--line);
  display: flex;
  flex-wrap: wrap;
  gap: 8px 18px;
  justify-content: space-between;
  font-size: .7rem;
  color: #9aa4b8;
}
.foot b { color: #64748b; font-weight: 600; }

/* Responsive ------------------------------------------------------------ */
@media (max-width: 960px) {
  .hero { padding: 26px 22px; border-radius: 20px; }
  [data-testid="stMainBlockContainer"], .block-container { padding-top: 1rem; }
  .hero h1 { font-size: 1.72rem; }
  .hero p { font-size: .88rem; }
  .hero-mark { display: none; }
  .result-band { font-size: 1.8rem; }
  .tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .prob-row { grid-template-columns: 82px 1fr 48px; }
  .cm-grid { grid-template-columns: 62px repeat(4, 1fr); gap: 4px; }
  .cm-cell { min-height: 46px; font-size: .85rem; }
  .section h2 { white-space: normal; }
  .section-note { display: none; }
}
@media (max-width: 560px) {
  .tiles { grid-template-columns: 1fr; }
  .result-card { padding: 20px 18px 16px; }
  .ring-wrap, .ring-wrap svg { width: 150px; height: 150px; }
  .weight-row { grid-template-columns: 92px 1fr 40px; }
}
</style>
"""


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def inject() -> None:
    """Write the stylesheet into the page."""
    st.markdown(CSS, unsafe_allow_html=True)


def h(markup: str) -> None:
    """Render a trusted HTML fragment."""
    st.markdown(markup, unsafe_allow_html=True)


def esc(value: object) -> str:
    return html.escape(str(value))


def tone(name: str) -> dict:
    return TONES.get(name, TONES["brand"])


def blend(color: str, other: str = "#ffffff", amount: float = 0.35) -> str:
    """Public wrapper around :func:`_mix`."""
    return _mix(color, other, amount)


def _mix(color: str, other: str = "#ffffff", amount: float = 0.35) -> str:
    """Blend two hex colours (simple, no extra dependency)."""
    a = color.lstrip("#")
    b = other.lstrip("#")
    if len(a) != 6 or len(b) != 6:
        return color
    out = []
    for i in (0, 2, 4):
        av = int(a[i : i + 2], 16)
        bv = int(b[i : i + 2], 16)
        out.append(round(av + (bv - av) * amount))
    return "#" + "".join(f"{v:02x}" for v in out)


# ---------------------------------------------------------------------------
# Hero and section headers
# ---------------------------------------------------------------------------
def hero(kicker: str, title: str, accent: str, subtitle: str, facts: list[tuple[str, str]]) -> None:
    """
    Page hero. ``title`` is the plain part of the headline and ``accent`` is
    rendered with the gradient treatment.
    """
    fact_html = "".join(
        f'<span class="fact"><b>{esc(value)}</b> {esc(label)}</span>' for value, label in facts
    )
    h(
        '<div class="hero">'
        '<div class="hero-mark" aria-hidden="true">'
        '<svg viewBox="0 0 300 300" fill="none">'
        '<circle cx="150" cy="150" r="118" stroke="rgba(199,210,254,.55)" stroke-width="1" stroke-dasharray="3 9"/>'
        '<circle cx="150" cy="150" r="86" stroke="rgba(103,232,249,.45)" stroke-width="1"/>'
        '<circle cx="150" cy="150" r="54" stroke="rgba(196,181,253,.5)" stroke-width="1" stroke-dasharray="6 6"/>'
        '<circle cx="150" cy="32" r="4.5" fill="#67e8f9"/>'
        '<circle cx="256" cy="196" r="4" fill="#a5b4fc"/>'
        '<circle cx="58" cy="222" r="3.5" fill="#c4b5fd"/>'
        '<path d="M150 64 L212 196 L88 196 Z" stroke="rgba(255,255,255,.22)" stroke-width="1"/>'
        "</svg>"
        "</div>"
        f'<span class="kicker"><span class="pulse"></span>{esc(kicker)}</span>'
        f'<h1>{esc(title)}<br><span class="grad">{esc(accent)}</span></h1>'
        f"<p>{esc(subtitle)}</p>"
        f'<div class="facts">{fact_html}</div>'
        "</div>"
    )


def section(title: str, icon: str = "", note: str = "") -> None:
    note_html = f'<span class="section-note">{esc(note)}</span>' if note else ""
    icon_html = f'<span class="section-icon">{icon}</span>' if icon else ""
    h(
        f'<div class="section">{icon_html}<h2>{esc(title)}</h2>'
        f'<span class="section-line"></span>{note_html}</div>'
    )


def footer(left: str, right: str = "") -> None:
    right_html = f"<span>{right}</span>" if right else ""
    h(f'<div class="foot"><span>{left}</span>{right_html}</div>')


# ---------------------------------------------------------------------------
# Stat tiles
# ---------------------------------------------------------------------------
def tiles(items: list[dict]) -> None:
    """
    A responsive strip of KPI tiles.

    Each item: ``label``, ``value``, ``hint``, optional ``icon`` and
    ``tone`` (a key of :data:`TONES`).
    """
    cells = []
    for item in items:
        theme = tone(item.get("tone", "brand"))
        icon = item.get("icon", "")
        icon_html = f'<span class="tile-ico">{icon}</span>' if icon else ""
        cells.append(
            f'<div class="tile" style="--tone:{theme["color"]};--tone-soft:{theme["soft"]}">'
            f'<div class="tile-top">{icon_html}<span class="tile-label">{esc(item["label"])}</span></div>'
            f'<div class="tile-value">{esc(item["value"])}</div>'
            f'<div class="tile-hint">{esc(item.get("hint", ""))}</div>'
            "</div>"
        )
    h(f'<div class="tiles">{"".join(cells)}</div>')


# ---------------------------------------------------------------------------
# Score ring
# ---------------------------------------------------------------------------
def score_ring(value: float, color: str, band_soft: str, caption: str = "readiness") -> str:
    """Conic-style SVG donut showing a 0-100 score."""
    value = max(0.0, min(100.0, float(value)))
    radius = 78.0
    circumference = 2 * math.pi * radius
    offset = circumference * (1 - value / 100.0)
    gid = "rg" + re.sub(r"[^0-9a-zA-Z]", "", color)
    light = _mix(color, "#ffffff", 0.32)
    return (
        '<div class="ring-wrap">'
        '<svg viewBox="0 0 184 184" role="img">'
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0%" stop-color="{color}"/><stop offset="100%" stop-color="{light}"/>'
        "</linearGradient></defs>"
        '<circle class="ring-track" cx="92" cy="92" r="78"/>'
        f'<circle class="ring-fill" cx="92" cy="92" r="78" stroke="url(#{gid})" '
        f'stroke-dasharray="{circumference:.1f}" stroke-dashoffset="{offset:.1f}"/>'
        "</svg>"
        '<div class="ring-center">'
        f'<div class="ring-value">{value:.1f}</div>'
        '<div class="ring-unit">of 100</div>'
        f'<div class="ring-label" style="--band:{color};--band-soft:{band_soft}">{esc(caption)}</div>'
        "</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Probability bars
# ---------------------------------------------------------------------------
def prob_bars(items: list[tuple[str, float, str]], predicted: str, band_color: str) -> None:
    """CSS-only horizontal bars: ``items`` are (label, percent, colour)."""
    rows = []
    for label, pct, color in items:
        width = max(float(pct), 0.8) if pct > 0 else 0.0
        active = " is-predicted" if label == predicted else ""
        rows.append(
            f'<div class="prob-row{active}" style="--bar:{color};--band:{band_color}">'
            f'<span class="prob-label">{esc(label)}</span>'
            '<span class="prob-track">'
            f'<span class="prob-fill" style="width:{width:.1f}%"></span>'
            "</span>"
            f'<span class="prob-val">{pct:.1f}%</span>'
            "</div>"
        )
    h(
        '<div class="panel">'
        '<div class="panel-head">'
        "<div>"
        '<div class="panel-title">How the tree voted</div>'
        '<div class="panel-sub">Share of training evidence behind each band</div>'
        "</div>"
        f'<span class="panel-note">{esc(predicted)}</span>'
        "</div>"
        f'<div class="prob-list">{"".join(rows)}</div>'
        "</div>"
    )


# ---------------------------------------------------------------------------
# Chips
# ---------------------------------------------------------------------------
def chips(items: list[dict]) -> str:
    """Small indicator pills: ``label``, ``value``, ``color``, optional ``icon``."""
    parts = []
    for item in items:
        icon = f'{item["icon"]} ' if item.get("icon") else ""
        parts.append(
            '<span class="chip">'
            f'<span class="dot" style="background:{item["color"]}"></span>'
            f'{esc(icon)}{esc(item["label"])}: '
            f'<b style="color:{item["color"]}">{esc(item["value"])}</b></span>'
        )
    return f'<div class="chips">{"".join(parts)}</div>'


# ---------------------------------------------------------------------------
# Advice cards and callouts
# ---------------------------------------------------------------------------
def tip_card(item: dict) -> str:
    """One improvement suggestion."""
    theme = tone(item.get("tone", "brand"))
    sub = item.get("sub", "")
    sub_html = f'<div class="tip-sub">{esc(sub)}</div>' if sub else ""
    return (
        f'<div class="tip" style="--tone:{theme["color"]};--tone-soft:{theme["soft"]}">'
        '<div class="tip-head">'
        f'<span class="tip-icon">{item.get("icon", "")}</span>'
        f'<div><div class="tip-area">{esc(item["area"])}</div>{sub_html}</div>'
        f'<span class="badge">{esc(item.get("badge", ""))}</span>'
        "</div>"
        f'<p class="tip-msg">{esc(item["message"])}</p>'
        '<div class="tip-action">'
        "<span>Do this</span>"
        f'{esc(item["action"])}'
        "</div>"
        "</div>"
    )


def callout(icon: str, title: str, body: str, tone_name: str = "brand", warn: bool = False) -> str:
    theme = tone(tone_name)
    extra = " warn" if warn else ""
    return (
        f'<div class="callout{extra}" style="--tone:{theme["color"]};--tone-soft:{theme["soft"]}">'
        f'<div class="ico">{icon}</div>'
        '<div>'
        f'<div class="ttl">{esc(title)}</div>'
        f'<div class="sub">{body}</div>'
        "</div>"
        "</div>"
    )


def softnote(icon: str, body: str) -> str:
    return f'<div class="softnote"><span class="i">{icon}</span><span>{body}</span></div>'


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
def empty_state(icon: str, title: str, body: str, hints: list[str] | None = None) -> None:
    hint_html = ""
    if hints:
        pills = "".join(f"<span>{esc(item)}</span>" for item in hints)
        hint_html = f'<div class="quick">{pills}</div>'
    h(
        '<div class="empty">'
        f'<div class="art">{icon}</div>'
        f"<h3>{esc(title)}</h3>"
        f"<p>{esc(body)}</p>"
        f"{hint_html}"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def table(
    headers: list[str],
    rows: list[list[str]],
    title: str = "",
    subtitle: str = "",
    note: str = "",
    numeric: list[bool] | None = None,
    highlight: list[int] | None = None,
    dot_colors: dict[str, str] | None = None,
) -> None:
    """
    Render a static HTML table inside a card.

    ``rows`` should already be formatted strings; they are escaped here.
    ``numeric`` marks which columns are right-aligned. ``dot_colors`` maps a
    cell value (for example a band name) to a colour dot.
    """
    numeric = numeric or [False] * len(headers)
    highlight = highlight or []
    head = "".join(
        f'<th class="{"num" if numeric[i] else ""}">{esc(name)}</th>'
        for i, name in enumerate(headers)
    )
    body = []
    for index, row in enumerate(rows):
        cls = ' class="hl"' if index in highlight else ""
        cells = []
        for i, value in enumerate(row):
            text = esc(value)
            if dot_colors and str(value) in dot_colors:
                text = (
                    f'<span class="dot" style="background:{dot_colors[str(value)]}"></span>{text}'
                )
            cells.append(f'<td class="{"num" if numeric[i] else ""}">{text}</td>')
        body.append(f"<tr{cls}>{''.join(cells)}</tr>")

    header_html = ""
    if title or subtitle or note:
        sub = f'<div class="panel-sub">{esc(subtitle)}</div>' if subtitle else ""
        note_html = f'<span class="panel-note">{esc(note)}</span>' if note else ""
        header_html = (
            '<div class="panel-head"><div>'
            f'<div class="panel-title">{esc(title)}</div>{sub}</div>'
            f"{note_html}</div>"
        )

    h(
        '<div class="panel tbl-panel">'
        f"{header_html}"
        f'<table class="tbl"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
        "</div>"
    )


# ---------------------------------------------------------------------------
# Small row lists and meters
# ---------------------------------------------------------------------------
def rowlist(items: list[dict]) -> str:
    """Compact list of indicators with a value and a status pill."""
    rows = []
    for item in items:
        status = item.get("status_label", "")
        color = item.get("color", BRAND)
        soft = item.get("soft") or _mix(color, "#ffffff", 0.9)
        pill = (
            f'<span class="s" style="color:{color};background:{soft}">{esc(status)}</span>'
            if status
            else ""
        )
        icon = f'{item.get("icon", "")} ' if item.get("icon") else ""
        rows.append(
            '<div class="rowitem">'
            f'<span class="k">{esc(icon)}{esc(item["label"])}</span>'
            f'<span class="v">{esc(item["value"])}</span>{pill}'
            "</div>"
        )
    return f'<div class="rowlist">{"".join(rows)}</div>'


def weight_meter(items: list[dict]) -> str:
    """Rows of label / contribution bar / value."""
    rows = []
    for item in items:
        theme = tone(item.get("tone", "brand"))
        share = max(0.0, min(100.0, float(item["share"])))
        rows.append(
            f'<div class="weight-row" style="--tone:{theme["color"]};--tone-soft:{theme["soft"]}">'
            f'<span class="weight-name">{esc(item["label"])}</span>'
            '<span class="weight-track">'
            f'<span class="weight-fill" style="width:{share:.1f}%"></span>'
            "</span>"
            f'<span class="weight-val">{esc(item["value"])}</span>'
            "</div>"
        )
    return f'<div class="weights">{"".join(rows)}</div>'


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def confusion_grid(
    matrix: list[list[int]],
    labels: list[str],
    total: int,
    title: str = "Confusion matrix",
    subtitle: str = "Rows are the true band, columns are the model's guess.",
) -> None:
    peak = max((max(row) for row in matrix), default=1) or 1
    cells = ['<div></div>']
    for label in labels:
        cells.append(f'<div class="cm-head">{esc(label)}</div>')

    for i, label in enumerate(labels):
        cells.append(f'<div class="cm-side">{esc(label)}</div>')
        row_total = sum(matrix[i]) or 1
        for j, value in enumerate(matrix[i]):
            strength = (value / peak) ** 0.72
            alpha = 0.05 + 0.78 * strength
            text = "#ffffff" if alpha > 0.48 else "#312e81"
            sub = "rgba(255,255,255,.78)" if alpha > 0.48 else "#8a94a8"
            cls = "cm-cell diag" if i == j else "cm-cell"
            cells.append(
                f'<div class="{cls}" style="background:rgba(79,70,229,{alpha:.2f});color:{text}">'
                f"{value}"
                f'<small style="color:{sub}">{value / row_total:.0%}</small>'
                "</div>"
            )

    h(
        '<div class="panel">'
        '<div class="panel-head"><div>'
        f'<div class="panel-title">{esc(title)}</div>'
        f'<div class="panel-sub">{esc(subtitle)}</div>'
        "</div>"
        f'<span class="panel-note">{total} students</span>'
        "</div>"
        f'<div class="cm-grid">{"".join(cells)}</div>'
        '<div class="cm-axes"><span>actual</span><span>predicted</span></div>'
        "</div>"
    )


# ---------------------------------------------------------------------------
# Sidebar fragments
# ---------------------------------------------------------------------------
def sidebar_brand(name: str, subtitle: str, logo: str = "🎓") -> None:
    h(
        '<div class="sb-brand">'
        f'<div class="sb-logo">{logo}</div>'
        f'<div><div class="sb-name">{esc(name)}</div><div class="sb-sub">{esc(subtitle)}</div></div>'
        "</div>"
    )


def sidebar_card(title: str, rows: list[tuple[str, str]]) -> None:
    row_html = "".join(
        f'<div class="sb-row"><span class="k">{esc(key)}</span><span class="v">{esc(value)}</span></div>'
        for key, value in rows
    )
    h(
        '<div class="sb-card">'
        f'<div class="sb-label" style="margin-top:0">{esc(title)}</div>'
        f"{row_html}"
        "</div>"
    )


def sidebar_steps(title: str, steps: list[str]) -> None:
    items = "".join(f"<li>{esc(step)}</li>" for step in steps)
    h(f'<div class="sb-label">{esc(title)}</div><ol class="steps">{items}</ol>')


# ---------------------------------------------------------------------------
# Plotly styling
# ---------------------------------------------------------------------------
def style_fig(
    fig: go.Figure,
    title: str = "",
    height: int | None = None,
    legend: bool = False,
) -> go.Figure:
    """Apply the shared chart look (fonts, spacing, grid, title)."""
    layout: dict = {
        "font": PLOT_FONT,
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "margin": {"l": 12, "r": 20, "t": 54 if title else 18, "b": 14},
        "showlegend": legend,
        "hoverlabel": {
            "bgcolor": "#ffffff",
            "bordercolor": LINE,
            "font": {"family": PLOT_FONT["family"], "size": 11, "color": "#334155"},
        },
    }
    if title:
        layout["title"] = {
            "text": f"<b>{title}</b>",
            "x": 0.012,
            "xanchor": "left",
            "y": 0.97,
            "yanchor": "top",
            "font": {"family": PLOT_FONT["family"], "size": 13.5, "color": INK},
        }
    if height:
        layout["height"] = height
    if legend:
        layout["legend"] = {
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.14,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 10.5},
        }
    fig.update_layout(**layout)
    fig.update_xaxes(
        gridcolor=GRID,
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont={"size": 10.5, "color": MUTED},
    )
    fig.update_yaxes(
        gridcolor=GRID,
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont={"size": 10.5, "color": MUTED},
    )
    return fig
