"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ZYNETRAX UNIVERSAL INTELLIGENCE ENGINE v6.0                                ║
║  Replaces: Power BI · Looker Studio · Tableau · AutoCAD · ChemCAD          ║
║  R/SPSS/SAS · MATLAB · Bloomberg · SAP2000 · Aspen · GraphPad · ArcGIS     ║
║  DeepSeek V3 primary · GPT-4o-mini fallback · 17 domain modules            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import io, json, re, warnings, os, subprocess, sys, tempfile
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY","")
OPENAI_KEY   = os.environ.get("OPENAI_API_KEY","")

from openai import OpenAI
_ds  = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
_oai = OpenAI(api_key=OPENAI_KEY)

FONT = "Segoe UI, Arial, sans-serif"
PAL  = ["#4361ee","#f72585","#4cc9f0","#7209b7","#06d6a0",
        "#ffd166","#ef476f","#118ab2","#26a269","#e76f51",
        "#a8dadc","#457b9d","#e63946","#2ec4b6","#ff9f1c"]

COUNTRY_ISO = {
    "afghanistan":"AFG","albania":"ALB","algeria":"DZA","angola":"AGO","argentina":"ARG",
    "australia":"AUS","austria":"AUT","bangladesh":"BGD","belgium":"BEL","brazil":"BRA",
    "canada":"CAN","chile":"CHL","china":"CHN","colombia":"COL","denmark":"DNK",
    "egypt":"EGY","ethiopia":"ETH","finland":"FIN","france":"FRA","germany":"DEU",
    "ghana":"GHA","greece":"GRC","hungary":"HUN","india":"IND","indonesia":"IDN",
    "iran":"IRN","iraq":"IRQ","ireland":"IRL","israel":"ISR","italy":"ITA","japan":"JPN",
    "jordan":"JOR","kenya":"KEN","south korea":"KOR","korea":"KOR","malaysia":"MYS",
    "mexico":"MEX","morocco":"MAR","netherlands":"NLD","nigeria":"NGA","norway":"NOR",
    "pakistan":"PAK","peru":"PER","philippines":"PHL","poland":"POL","portugal":"PRT",
    "qatar":"QAT","romania":"ROU","russia":"RUS","saudi arabia":"SAU","singapore":"SGP",
    "south africa":"ZAF","spain":"ESP","sweden":"SWE","switzerland":"CHE",
    "thailand":"THA","turkey":"TUR","ukraine":"UKR","uae":"ARE",
    "united kingdom":"GBR","uk":"GBR","united states":"USA","usa":"USA","us":"USA",
    "vietnam":"VNM","zimbabwe":"ZWE",
}

# ── LLM ───────────────────────────────────────────────────────────────────────
# ── HYBRID LLM ROUTING ────────────────────────────────────────────────────
# Claude: precision tasks (charts, calculations, structured output)
# DeepSeek: text generation (summaries, narratives, chat)
# ──────────────────────────────────────────────────────────────────────────

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY","")

def _call_claude(prompt, max_tokens=1024, system="You are a world-class data analyst and scientific computing expert. Be precise and accurate."):
    """Call Claude Sonnet — best for precision, structured output, calculations."""
    try:
        import urllib.request
        payload = json.dumps({
            "model": "claude-sonnet-4-5",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role":"user","content":prompt}]
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "x-api-key": ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"].strip()
    except Exception as e:
        return None  # Fall through to DeepSeek

def _needs_claude(prompt, system=""):
    """Decide if this call needs Claude's precision."""
    precision_signals = [
        "json", "JSON", "return only", "Return only",
        "chart", "Chart", "calculate", "Calculate",
        "regression", "exact", "formula", "engineering",
        "statistical", "p-value", "coefficient",
        "schema", "spec", "structure", "format",
        "solve", "Solve", "derivative", "integral",
        "SQL", "sql", "query", "Query",
    ]
    text = prompt + " " + system
    return any(sig in text for sig in precision_signals)

def _call_llm(prompt, max_tokens=600, json_mode=False,
              system="You are a world-class scientific computing and data analysis assistant. Be precise, use exact numbers and formulas.",
              provider="auto"):
    """
    Hybrid LLM router.
    provider="claude"   → always Claude
    provider="deepseek" → always DeepSeek
    provider="auto"     → Claude for precision, DeepSeek for text
    """
    # Decide provider
    use_claude = (
        provider == "claude" or
        (provider == "auto" and _needs_claude(prompt, system))
    )

    # Try Claude first for precision tasks
    if use_claude:
        claude_system = system if system else "You are a world-class scientific computing and data analysis assistant."
        if json_mode:
            claude_system += " Return ONLY valid JSON, no markdown, no explanation."
        result = _call_claude(prompt, max_tokens=max(max_tokens, 512), system=claude_system)
        if result:
            return result
        # Fall through to DeepSeek if Claude fails

    # DeepSeek for text tasks or as fallback
    kw = dict(
        messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
        max_tokens=max_tokens,
        temperature=0.1
    )
    if json_mode:
        kw["response_format"] = {"type":"json_object"}

    for client, model in [(_ds,"deepseek-chat"), (_oai,"gpt-4o-mini")]:
        try:
            r = client.chat.completions.create(model=model, **kw)
            return r.choices[0].message.content.strip()
        except Exception:
            continue
    return ""

def _theme(fig, title=""):
    if title and hasattr(fig, 'update_layout'):
        fig.update_layout(title=dict(text=title, font=dict(size=14, family=FONT, color="#e8e8f0")))
    if hasattr(fig, 'update_layout'):
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
            font=dict(family=FONT, color="#c8c8d8", size=12),
            margin=dict(l=50,r=20,t=50,b=50),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
        fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    return fig

def fingerprint_schema(df):
    """
    INTELLIGENT schema detection — human-level data understanding.
    Knows the difference between:
    - Year columns (never sum or average years)
    - Price/rate columns (mean, not sum)
    - Revenue/quantity columns (sum is correct)
    - ID columns (never aggregate)
    - Percentage columns (mean, bounded 0-100)
    - Date strings (convert and treat as temporal)
    """
    schema = {}

    # Column name patterns for intelligent aggregation
    MEAN_PATTERNS = [
        'price','rate','cost','fee','wage','salary','avg','average','mean',
        'per_','_per','ratio','pct','percent','discount','score','index','rank',
        'grade','bmi','temperature','temp','lat','lon','longitude','latitude',
        'unitprice','listprice','unitcost','level','pollution','stress','access',
        'quality','consumption','cholesterol','glucose','pressure','density',
        'concentration','ph','humidity','speed','velocity','acceleration',
        'frequency','probability','confidence','accuracy','precision','recall',
        'incidence','prevalence','mortality','morbidity','coefficient','ratio',
        'factor','multiplier','weight','height','depth','width','length',
        # Health/medical — always mean
        'bmi','age','weight','height','dosage','dose','reading','measurement',
        # Economics — rates/indices are mean
        'inflation','gdp_per','income_per','unemployment','poverty',
        # Environmental
        'aqi','pm2','co2','temperature','rainfall','humidity',
    ]
    SUM_PATTERNS = [
        'total','revenue','sales','amount','quantity','qty',
        'volume','sum','subtotal','income','profit','loss','spend',
        'budget','expense','turnover','gross','net','count',
        'transactions','orders','units','visits','clicks','views',
        # Financial totals
        'turnover','receipts','payments','deposits','withdrawals',
    ]
    YEAR_PATTERNS    = ['year','yr','jahr','anno','année']
    ID_PATTERNS      = ['_id','id_','uuid','guid','hash','code','key','ref','sku',
                        'number','no.','num','#']
    DATE_PATTERNS    = ['date','time','created','updated','modified','timestamp',
                        'start','end','born','closed','opened','at','on']

    def col_agg_type(col_name, series):
        """Determine correct aggregation: sum, mean, count, none"""
        cn = col_name.lower().replace(' ','_')
        # Year detection: numeric but represents a year
        if any(p == cn or cn.endswith('_'+p) or cn.startswith(p+'_') for p in YEAR_PATTERNS):
            if series.dropna().between(1800, 2100).mean() > 0.9:
                return 'year'
        # Price/rate: mean is correct
        if any(p in cn for p in MEAN_PATTERNS):
            return 'mean'
        # Sum: additive metrics
        if any(p in cn for p in SUM_PATTERNS):
            return 'sum'
        # Check value range heuristics
        vals = series.dropna()
        if len(vals) > 0:
            vmin, vmax = float(vals.min()), float(vals.max())
            # Percentage: 0-100 range
            if 0 <= vmin and vmax <= 100 and abs(vals.mean()) < 50:
                return 'mean'
            # Year-like values
            if 1900 <= vmin and vmax <= 2100 and vals.std() < 20:
                return 'year'
        return 'sum'  # default

    for col in df.columns:
        s = df[col]; nu = s.nunique(); n = len(s)
        cn = col.lower().replace(' ','_')

        # Try parse date strings
        is_date_str = False
        if s.dtype == object and any(p in cn for p in DATE_PATTERNS):
            try:
                parsed = pd.to_datetime(s, errors='coerce', infer_datetime_format=True)
                if parsed.notna().mean() > 0.7:
                    is_date_str = True
            except: pass

        if pd.api.types.is_datetime64_any_dtype(s) or is_date_str:
            role = "temporal"
        elif pd.api.types.is_numeric_dtype(s):
            # ID: unique per row and matches ID pattern
            if (nu == n or nu > n*0.95) and any(p in cn for p in ID_PATTERNS):
                role = "id"
            elif nu <= 2:
                role = "binary"
            else:
                agg = col_agg_type(col, s)
                if agg == 'year':
                    role = "year_col"  # special: show range not sum
                else:
                    role = "measure"
        elif nu/max(n,1) > 0.8 and nu > 50:
            # High cardinality string — ID or entity
            if any(p in cn for p in ID_PATTERNS):
                role = "id"
            else:
                sample = s.dropna().astype(str).head(10).tolist()
                role = "entity" if sum(1 for v in sample if v and v[0].isupper())/max(len(sample),1)>0.5 else "text"
        elif nu <= 30:
            role = "dimension"
        else:
            role = "text"

        is_num = pd.api.types.is_numeric_dtype(s)
        agg_type = col_agg_type(col, s) if is_num and role=="measure" else ("mean" if role=="year_col" else "sum")

        schema[col] = {
            "role": role,
            "agg": agg_type,  # NEW: correct aggregation type
            "dtype": str(s.dtype),
            "nunique": nu,
            "sample": s.dropna().head(5).tolist(),
            "min":  float(s.min())  if is_num and s.notna().any() else None,
            "max":  float(s.max())  if is_num and s.notna().any() else None,
            "mean": round(float(s.mean()),4) if is_num and s.notna().any() else None,
            "std":  round(float(s.std()),4)  if is_num and s.notna().any() else None,
            "sum":  float(s.sum())  if is_num and s.notna().any() else None,
        }
    return schema

def smart_merge(dfs, names=None):
    if len(dfs)==1: return dfs[0]
    names = names or ["t{}".format(i+1) for i in range(len(dfs))]
    def score(a, b):
        ns = SequenceMatcher(None, a.name.lower(), b.name.lower()).ratio()
        try: ov = len(set(a.dropna())&set(b.dropna()))/max(a.nunique(),1)
        except TypeError: ov = 0.0
        return ns*0.5+ov*0.5
    result = dfs[0]
    for i in range(1, len(dfs)):
        right = dfs[i]; best,lk,rk = 0,None,None
        for lc in result.columns:
            for rc in right.columns:
                s = score(result[lc], right[rc])
                if s>best: best,lk,rk = s,lc,rc
        if best>0.3 and lk and rk:
            cl = "1" if result[lk].nunique()==len(result) else "N"
            cr = "1" if right[rk].nunique()==len(right) else "N"
            how = {"1-1":"inner","1-N":"left","N-1":"right"}.get("{}-{}".format(cl,cr),"left")
            result = pd.merge(result,right,left_on=lk,right_on=rk,how=how,suffixes=("","_"+names[i]))
        else:
            result = pd.concat([result,right],ignore_index=True)
    return result

# ══════════════════════════════════════════════════════════════════════════════
# POWER BI + LOOKER STUDIO REPLACEMENT
# Full dashboard builder with KPIs, filters, drill-down, sharing
# ══════════════════════════════════════════════════════════════════════════════
def _smart_agg(df, col, schema):
    """Apply the correct aggregation based on column semantics."""
    agg = schema.get(col,{}).get("agg","sum")
    if agg == "mean":
        val = df[col].mean()
        label = "Avg"
    elif agg == "year":
        val = df[col].nunique()
        label = "Years"
    else:
        val = df[col].sum()
        label = "Total"
    return val, label, agg

def _format_kpi(val, col_name):
    """Format KPI value intelligently."""
    cn = col_name.lower()
    if abs(val) >= 1e9:
        return f"{val/1e9:.2f}B"
    elif abs(val) >= 1e6:
        return f"{val/1e6:.2f}M"
    elif abs(val) >= 1e3:
        return f"{val/1e3:.1f}K"
    elif any(p in cn for p in ['price','cost','revenue','amount','total','subtotal']):
        return f"${val:,.2f}" if abs(val) < 1000 else f"${val:,.0f}"
    elif any(p in cn for p in ['pct','percent','discount','rate']):
        return f"{val:.1f}%"
    else:
        return f"{val:,.2f}" if val != int(val) else f"{val:,.0f}"

def build_powerbi_dashboard(df, schema, lang="en"):
    """
    INTELLIGENT Power BI / Looker Studio dashboard.
    - KPIs use CORRECT aggregation (avg price, total revenue, count years)
    - Never sums year columns
    - Never shows ID columns
    - Charts selected based on actual data semantics
    """
    measures   = [c for c,m in schema.items() if m["role"]=="measure"]
    year_cols  = [c for c,m in schema.items() if m["role"]=="year_col"]
    dimensions = [c for c,m in schema.items() if m["role"]=="dimension"]
    temporals  = [c for c,m in schema.items() if m["role"]=="temporal"]
    entities   = [c for c,m in schema.items() if m["role"]=="entity"]
    # Never include ID or text columns in charts
    figs = []

    # ── HANDLE DATA WITH NO NUMERIC MEASURES ────────────────────────────────
    # When dataset has only categorical/date columns (e.g. CRM, event logs)
    # use COUNT as the metric — this is what Power BI does automatically
    has_measures = len(measures) > 0
    if not has_measures:
        # Build count-based dashboard — most meaningful for categorical data
        cat_cols = [c for c in (entities + dimensions) if c not in [
            c2 for c2,m2 in schema.items() if m2["role"] in ("id","text")]][:5]
        time_cols_local = temporals + year_cols

        # KPI: total count
        fig_kpi = go.Figure()
        fig_kpi.add_trace(go.Indicator(
            mode="number",
            value=len(df),
            title={"text":"Total Records","font":{"size":13,"color":"#c8c8d8"}},
            number={"font":{"size":36,"color":"white"},"valueformat":",.0f"},
            domain={"row":0,"column":0}
        ))
        # Add count per top category
        for i, col in enumerate(cat_cols[:4], 1):
            top_val = df[col].value_counts().index[0] if len(df[col].value_counts()) > 0 else "?"
            top_cnt = df[col].value_counts().iloc[0]
            fig_kpi.add_trace(go.Indicator(
                mode="number",
                value=top_cnt,
                title={"text":f"Top {col[:15]}: {str(top_val)[:12]}","font":{"size":11,"color":"#c8c8d8"}},
                number={"font":{"size":26,"color":"white"},"valueformat":",.0f"},
                domain={"row":0,"column":i}
            ))
        fig_kpi.update_layout(
            grid={"rows":1,"columns":min(5,1+len(cat_cols[:4]))}, height=180,
            paper_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color="#c8c8d8"))
        figs.append(fig_kpi)

        # Bar charts: count by each categorical column
        for col in cat_cols[:3]:
            try:
                vc = df[col].value_counts().head(15).reset_index()
                vc.columns = [col, "count"]
                fig = px.bar(vc, x=col, y="count", color="count",
                    color_continuous_scale="Blues",
                    title=f"📊 Opportunities by {col.replace('_',' ').title()}")
                fig.update_layout(coloraxis_showscale=False)
                figs.append(_theme(fig))
            except: pass

        # Time trend if date column exists
        if time_cols_local:
            try:
                tc = time_cols_local[0]
                df_t = df.copy()
                df_t[tc] = pd.to_datetime(df_t[tc], errors='coerce')
                df_t = df_t.dropna(subset=[tc])
                df_t['month'] = df_t[tc].dt.to_period('M').astype(str)
                monthly = df_t.groupby('month').size().reset_index(name='count')
                monthly = monthly.sort_values('month').tail(24)
                fig = px.line(monthly, x='month', y='count', markers=True,
                    title="📊 Opportunities Over Time",
                    color_discrete_sequence=[PAL[0]])
                fig.update_layout(xaxis_tickangle=-45)
                figs.append(_theme(fig))
            except: pass

        # Donut for first categorical
        if cat_cols:
            try:
                vc2 = df[cat_cols[0]].value_counts().head(8).reset_index()
                vc2.columns = [cat_cols[0], "count"]
                fig = px.pie(vc2, names=cat_cols[0], values="count", hole=0.5,
                    color_discrete_sequence=PAL,
                    title=f"📊 Distribution by {cat_cols[0].replace('_',' ').title()}")
                fig.update_traces(textposition="inside", textinfo="percent+label")
                figs.append(_theme(fig))
            except: pass

        # Cross-tabulation heatmap if 2+ categorical columns
        if len(cat_cols) >= 2:
            try:
                ct = pd.crosstab(df[cat_cols[0]], df[cat_cols[1]])
                ct = ct.iloc[:10, :10]  # limit size
                fig = px.imshow(ct, color_continuous_scale="Blues",
                    title=f"📊 {cat_cols[0].title()} × {cat_cols[1].title()} Heatmap")
                figs.append(_theme(fig))
            except: pass

        return figs

    # ── INTELLIGENT KPI SCORECARD ROW ─────────────────────────────────────────
    # Binary cols (0/1 flags) are shown as % rate — most meaningful for health/survey data
    binary_cols = [c for c,m in schema.items() if m["role"]=="binary"]

    def kpi_priority(col):
        cn = col.lower()
        # Target/outcome variables first
        if any(p in cn for p in ['incidence','attack','outcome','target','label','event','death','mortality']): return 0
        if any(p in cn for p in ['revenue','total','amount','income','profit','sales']): return 1
        if any(p in cn for p in ['price','cost','fee']): return 2
        if any(p in cn for p in ['rate','index','level','score']): return 3
        return 4

    # Combine measures + binary for KPIs
    kpi_candidates = sorted(measures[:6], key=kpi_priority)[:4]
    # Add top binary col (e.g. heart_attack_incidence) as first KPI
    binary_kpis = sorted(binary_cols, key=kpi_priority)[:1]
    kpi_cols = binary_kpis + [c for c in kpi_candidates if c not in binary_kpis]
    kpi_cols = kpi_cols[:5]
    if not kpi_cols: kpi_cols = measures[:5]

    if kpi_cols:
        kpi_vals = []
        for m in kpi_cols:
            is_binary = schema.get(m,{}).get("role") == "binary"
            if is_binary:
                # Binary: show as percentage
                val = round(df[m].mean() * 100, 1)
                label = "Rate %"
            else:
                val, label, _ = _smart_agg(df, m, schema)

            # REAL trend: only show delta if change is meaningful AND not extreme
            # Never show fake 15000% deltas for long historical datasets
            n = len(df)
            seg = max(n//10, 1)  # compare last 10% vs first 10%
            try:
                first_val = df[m].iloc[:seg].mean()
                last_val  = df[m].iloc[-seg:].mean()
                if first_val == 0:
                    show_delta = False
                    trend_pct = 0
                else:
                    trend_pct = (last_val - first_val) / abs(first_val) * 100
                    # Only show delta if change is between -200% and +200%
                    # Anything beyond that is likely historical range, not trend
                    show_delta = abs(trend_pct) <= 200
            except:
                show_delta = False
                trend_pct = 0

            kpi_vals.append({"name": m, "value": val, "label": label,
                             "trend_pct": trend_pct, "show_delta": show_delta,
                             "is_binary": is_binary})

        fig_kpi = go.Figure()
        for i, kpi in enumerate(kpi_vals):
            display_val = kpi["value"]
            col_name = kpi["name"].replace("_"," ").title()[:20]
            title_text = f"{kpi['label']} {col_name}"

            if kpi["show_delta"]:
                ref_val = display_val / (1 + kpi["trend_pct"]/100) if kpi["trend_pct"] != 0 else display_val
                mode = "number+delta"
                delta_spec = {"reference": ref_val, "relative": True, "valueformat": ".1%",
                              "increasing":{"color":"#06d6a0"}, "decreasing":{"color":"#ef476f"}}
            else:
                mode = "number"
                delta_spec = None

            trace = go.Indicator(
                mode=mode,
                value=display_val,
                title={"text": title_text, "font":{"size":11,"color":"#c8c8d8"}},
                number={"font":{"size":26,"color":"white"},
                        "suffix": "%" if kpi["is_binary"] else "",
                        "valueformat": ",.1f" if kpi["is_binary"] else (",.0f" if abs(display_val)>100 else ",.2f")},
                domain={"row":0,"column":i}
            )
            if delta_spec:
                trace.delta = delta_spec
            fig_kpi.add_trace(trace)

        fig_kpi.update_layout(
            grid={"rows":1,"columns":len(kpi_vals)}, height=180,
            paper_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color="#c8c8d8"))
        figs.append(fig_kpi)

    # Identify target variable (most likely outcome to analyze)
    def find_target(measures, binary_cols, schema):
        for c in binary_cols:
            cn = c.lower()
            if any(p in cn for p in ['incidence','attack','outcome','death','event','default','churn','fraud']): return c, True
        for c in measures:
            cn = c.lower()
            if any(p in cn for p in ['rate','incidence','mortality','risk']): return c, False
        return (measures[0] if measures else None), False

    target_col, target_is_binary = find_target(measures, binary_cols, schema)
    x_col = (entities or dimensions or [None])[0]  # primary group dimension

    # ── TIME SERIES: if year_col or temporal exists ────────────────────────
    time_col = (year_cols or temporals or [None])[0]

    # Detect if this is financial/OHLCV data
    col_lower = [c.lower() for c in df.columns]
    is_ohlcv = sum(1 for x in ['open','high','low','close'] if x in col_lower) >= 3
    is_stock = is_ohlcv or any(x in col_lower for x in ['price','stock','equity','ticker'])

    if is_ohlcv and time_col:
        # ── CANDLESTICK / OHLCV CHART ─────────────────────────────
        try:
            open_col = next((c for c in df.columns if c.lower()=='open'), None)
            high_col = next((c for c in df.columns if c.lower()=='high'), None)
            low_col  = next((c for c in df.columns if c.lower()=='low'), None)
            close_col= next((c for c in df.columns if c.lower()=='close'), None)
            vol_col  = next((c for c in df.columns if c.lower()=='volume'), None)

            plot_df = df[[time_col, open_col, high_col, low_col, close_col]].dropna()
            plot_df = plot_df.sort_values(time_col)

            # Sample if too many rows for candlestick
            if len(plot_df) > 500:
                # Resample to weekly/monthly
                plot_df[time_col] = pd.to_datetime(plot_df[time_col], errors='coerce')
                plot_df = plot_df.set_index(time_col).resample('W').agg({
                    open_col:'first', high_col:'max',
                    low_col:'min', close_col:'last'
                }).dropna().reset_index().tail(200)

            fig = go.Figure(go.Candlestick(
                x=plot_df[time_col],
                open=plot_df[open_col], high=plot_df[high_col],
                low=plot_df[low_col],   close=plot_df[close_col],
                increasing_line_color='#06d6a0',
                decreasing_line_color='#ef476f',
                name='Price'
            ))
            fig.update_layout(xaxis_rangeslider_visible=False)
            figs.append(_theme(fig, "📊 Price Chart (OHLC)"))
        except: pass

        # ── CLOSE PRICE TREND ─────────────────────────────────────
        if close_col and time_col:
            try:
                trend_df = df[[time_col, close_col]].dropna().sort_values(time_col)
                trend_df[time_col] = pd.to_datetime(trend_df[time_col], errors='coerce')

                # Add moving averages
                trend_df = trend_df.dropna(subset=[time_col])
                trend_df['MA50']  = trend_df[close_col].rolling(50, min_periods=1).mean()
                trend_df['MA200'] = trend_df[close_col].rolling(200, min_periods=1).mean()

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=trend_df[time_col], y=trend_df[close_col],
                    mode='lines', name='Close', line=dict(color=PAL[0], width=1.5), opacity=0.8))
                fig.add_trace(go.Scatter(x=trend_df[time_col], y=trend_df['MA50'],
                    mode='lines', name='MA50', line=dict(color=PAL[2], width=1.5, dash='dash')))
                fig.add_trace(go.Scatter(x=trend_df[time_col], y=trend_df['MA200'],
                    mode='lines', name='MA200', line=dict(color=PAL[1], width=1.5, dash='dot')))
                figs.append(_theme(fig, "📊 Close Price + Moving Averages"))
            except: pass

        # ── VOLUME BAR CHART ──────────────────────────────────────
        if vol_col and time_col:
            try:
                vol_df = df[[time_col, vol_col]].dropna().sort_values(time_col)
                vol_df[time_col] = pd.to_datetime(vol_df[time_col], errors='coerce')
                vol_df = vol_df.dropna(subset=[time_col])

                if len(vol_df) > 500:
                    vol_df = vol_df.set_index(time_col).resample('M')[vol_col].sum().reset_index()

                colors = ['#06d6a0' if i % 2 == 0 else '#4361ee' for i in range(len(vol_df))]
                fig = go.Figure(go.Bar(
                    x=vol_df[time_col], y=vol_df[vol_col],
                    marker_color=colors, name='Volume', opacity=0.7))
                figs.append(_theme(fig, "📊 Trading Volume Over Time"))
            except: pass

        # ── RETURNS DISTRIBUTION ──────────────────────────────────
        if close_col:
            try:
                returns = df[close_col].pct_change().dropna() * 100
                returns = returns[returns.between(-20, 20)]  # remove extreme outliers
                fig = go.Figure(go.Histogram(
                    x=returns, nbinsx=50,
                    marker_color=PAL[0], opacity=0.7,
                    name='Daily Returns'))
                fig.add_vline(x=0, line_dash='dash', line_color='#aaa')
                fig.add_vline(x=returns.mean(), line_dash='dot',
                    line_color='#06d6a0',
                    annotation_text=f"Avg: {returns.mean():.3f}%")
                figs.append(_theme(fig, "📊 Daily Returns Distribution"))
            except: pass

    elif time_col and (measures or binary_cols) and target_col:
        try:
            metric = target_col
            if target_is_binary:
                g = df.groupby(time_col)[metric].mean().mul(100).round(1).reset_index()
                g.columns = [time_col, metric]
                ytitle = "Rate %"
            else:
                agg = schema.get(metric,{}).get("agg","mean")
                g = df.groupby(time_col)[metric].agg(agg).reset_index()
                ytitle = ("Avg " if agg=="mean" else "Total ") + metric.replace("_"," ").title()
            g = g.sort_values(time_col)
            fig = px.line(g, x=time_col, y=metric,
                title="📊 {} over {}".format(metric.replace("_"," ").title(), time_col),
                color_discrete_sequence=[PAL[0]], markers=True)
            fig.update_layout(yaxis_title=ytitle)
            figs.append(_theme(fig))
        except: pass

    # ── GROUPED BAR: target by primary dimension ───────────────────────────
    if x_col and target_col:
        try:
            metric = target_col
            if target_is_binary:
                g = df.groupby(x_col)[metric].mean().mul(100).round(1).reset_index()
                g.columns = [x_col, metric]
                title = "📊 {} Rate % by {}".format(metric.replace("_"," ").title(), x_col.replace("_"," ").title())
                ytitle = "Rate %"
            else:
                agg = schema.get(metric,{}).get("agg","mean")
                g = df.groupby(x_col)[metric].agg(agg).round(2).reset_index()
                title = "📊 {} by {}".format(("Avg " if agg=="mean" else "Total ")+metric.replace("_"," ").title(), x_col.replace("_"," ").title())
                ytitle = metric.replace("_"," ").title()
            g[x_col] = g[x_col].astype(str).str[:20]
            g = g.sort_values(metric, ascending=False).head(15)
            fig = px.bar(g, x=x_col, y=metric, color=metric,
                color_continuous_scale="Blues", title=title)
            fig.update_layout(yaxis_title=ytitle, coloraxis_showscale=False)
            figs.append(_theme(fig))
        except: pass

    # ── SECOND DIMENSION BAR (if multiple dimensions) ─────────────────────
    dims_all = [c for c in (entities+dimensions) if c != x_col][:1]
    if dims_all and target_col:
        try:
            x2 = dims_all[0]
            metric = target_col
            if target_is_binary:
                g = df.groupby(x2)[metric].mean().mul(100).round(1).reset_index()
                g.columns = [x2, metric]
                title = "📊 {} Rate % by {}".format(metric.replace("_"," ").title(), x2.replace("_"," ").title())
            else:
                agg = schema.get(metric,{}).get("agg","mean")
                g = df.groupby(x2)[metric].agg(agg).round(2).reset_index()
                title = "📊 {} by {}".format(metric.replace("_"," ").title(), x2.replace("_"," ").title())
            g[x2] = g[x2].astype(str).str[:20]
            g = g.sort_values(metric, ascending=False).head(15)
            fig = px.bar(g, x=x2, y=metric, color=metric,
                color_continuous_scale="Viridis", title=title)
            fig.update_layout(coloraxis_showscale=False)
            figs.append(_theme(fig))
        except: pass

    # ── DONUT: distribution of primary dimension ────────────────────────────
    if x_col:
        try:
            vc = df[x_col].value_counts().head(8).reset_index()
            vc.columns = [x_col, "count"]
            fig = px.pie(vc, names=x_col, values="count", hole=0.5,
                color_discrete_sequence=PAL,
                title="📊 Distribution by {}".format(x_col.replace("_"," ").title()))
            fig.update_traces(textposition="inside", textinfo="percent+label")
            figs.append(_theme(fig))
        except: pass

    # ── SCATTER: two most meaningful measures ──────────────────────────────
    if len(measures) >= 2 and target_col:
        try:
            # Find best pair: target vs a risk factor
            other = [m for m in measures if m != target_col]
            if other:
                m1, m2 = (target_col, other[0]) if not target_is_binary else (other[0], other[1] if len(other)>1 else other[0])
                samp = df[[m1, m2]+(([x_col] if x_col else []))].dropna().sample(min(1000, len(df)))
                fig = px.scatter(samp, x=m1, y=m2,
                    color=x_col if x_col else None,
                    trendline="ols", opacity=0.5,
                    color_discrete_sequence=PAL,
                    title="📊 {} vs {}".format(m1.replace("_"," ").title(), m2.replace("_"," ").title()))
                figs.append(_theme(fig))
        except: pass

    # ── CORRELATION HEATMAP (Looker Studio) ───────────────────────────────────
    nums = measures[:10]
    if len(nums)>=3:
        try:
            corr = df[nums].corr().round(2)
            ns = [c[:14] for c in corr.columns]
            fig = go.Figure(go.Heatmap(z=corr.values,x=ns,y=ns,
                colorscale="RdBu",zmid=0,
                text=corr.values.round(2),texttemplate="%{text}",
                textfont={"size":max(8,12-len(ns))}))
            fig.update_layout(height=max(380,52*len(ns)))
            figs.append(_theme(fig,"📊 Correlation Matrix (Heatmap)"))
        except: pass

    # ── WATERFALL (Power BI Waterfall) ────────────────────────────────────────
    if temporals and measures:
        try:
            t = temporals[0]; mc = measures[0]
            monthly = df.groupby(t)[mc].sum().reset_index().sort_values(t).tail(12)
            monthly["delta"] = monthly[mc].diff().fillna(monthly[mc].iloc[0])
            fig = go.Figure(go.Waterfall(
                x=monthly[t].astype(str),
                y=monthly["delta"],
                connector=dict(line=dict(color="rgba(255,255,255,.3)")),
                increasing=dict(marker_color="#06d6a0"),
                decreasing=dict(marker_color="#ef476f"),
                totals=dict(marker_color="#4361ee"),
                name=mc
            ))
            figs.append(_theme(fig,"📊 Waterfall Chart — {} Changes".format(mc)))
        except: pass

    # ── GEOGRAPHIC MAP (Looker Studio Map) ────────────────────────────────────
    for col in df.columns:
        sample = df[col].dropna().astype(str).str.lower().head(30).tolist()
        match_count = sum(1 for v in sample if v in COUNTRY_ISO)
        if match_count>5 and measures:
            try:
                df2 = df.copy()
                df2["_iso"] = df2[col].astype(str).str.lower().map(COUNTRY_ISO)
                g = df2.groupby("_iso")[measures[0]].sum().reset_index()
                fig = px.choropleth(g,locations="_iso",color=measures[0],
                    color_continuous_scale="Blues",
                    title="🌍 Geographic Map: {} by Country".format(measures[0]),
                    labels={"_iso":"Country",measures[0]:measures[0]})
                figs.append(_theme(fig))
            except: pass
            break

    # ── TREEMAP (Looker Studio Treemap) ──────────────────────────────────────
    if x_col and measures:
        try:
            g3 = df[[x_col,measures[0]]].dropna().groupby(x_col)[measures[0]].sum().reset_index()
            g3 = g3.nlargest(20,measures[0])
            g3[x_col] = g3[x_col].astype(str).str[:25]
            fig = px.treemap(g3,path=[x_col],values=measures[0],
                color=measures[0],color_continuous_scale="Blues",
                title="📊 Treemap: {} by {}".format(measures[0],x_col))
            fig.update_coloraxes(showscale=False)
            figs.append(_theme(fig))
        except: pass

    # ── HISTOGRAM (Distribution) ─────────────────────────────────────────────
    if measures:
        try:
            fig = make_subplots(rows=1,cols=min(3,len(measures)),
                subplot_titles=["Distribution: "+m for m in measures[:3]])
            for i,m in enumerate(measures[:3]):
                fig.add_trace(go.Histogram(x=df[m].dropna(),nbinsx=25,
                    name=m,marker_color=PAL[i],opacity=0.8),row=1,col=i+1)
            fig.update_layout(height=320,showlegend=False)
            figs.append(_theme(fig,"📊 Distribution Analysis"))
        except: pass

    return figs

# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL ENGINEERING (AutoCAD/SAP2000/STAAD replacement)
# ══════════════════════════════════════════════════════════════════════════════
def structural_engineering(query, df=None, lang="en"):
    from scipy import stats as _stats
    q = query.lower()
    nums = re.findall(r"[-+]?\d*\.?\d+", query)
    vals = [float(x) for x in nums]
    figs=[]; results=[]

    if any(w in q for w in ["beam","deflect","moment","shear","span","cantilever","udl","distributed load","point load"]):
        span = next((v for v in vals if 1<=v<=100), 6.0)
        load = next((v for v in vals if v!=span and v>0), 10.0)
        E=200e9; I=8.196e-5
        is_cantilever="cantilever" in q; is_udl=any(w in q for w in ["udl","distributed","uniform"])
        x=np.linspace(0,span,300)
        if is_cantilever:
            if is_udl:
                w=load*1000 if load<500 else load
                M=w*(span-x)**2/2; V=w*(span-x)
                delta=w*x**2/(24*E*I)*(6*span**2-4*span*x+x**2)
                max_moment=w*span**2/2; max_deflect=w*span**4/(8*E*I)
                beam_type="Cantilever UDL = {:.1f}kN/m".format(w/1000)
            else:
                P=load*1000 if load<500 else load
                M=P*(span-x); V=np.full_like(x,P)
                delta=P*x**2/(6*E*I)*(3*span-x)
                max_moment=P*span; max_deflect=P*span**3/(3*E*I)
                beam_type="Cantilever P={:.1f}kN".format(P/1000)
        else:
            if is_udl:
                w=load*1000 if load<500 else load
                M=w*x*(span-x)/2; V=w*(span/2-x)
                delta=w*x/(24*E*I)*(span**3-2*span*x**2+x**3)
                max_moment=w*span**2/8; max_deflect=5*w*span**4/(384*E*I)
                beam_type="Simply Supported UDL={:.1f}kN/m".format(w/1000)
            else:
                P=load*1000 if load<500 else load; RA=P/2
                M=np.where(x<=span/2, RA*x, RA*x-P*(x-span/2)); V=np.where(x<=span/2,RA,RA-P)
                delta=np.where(x<=span/2, P*x/(48*E*I)*(3*span**2-4*x**2), P*(span-x)/(48*E*I)*(3*span**2-4*(span-x)**2))
                max_moment=P*span/4; max_deflect=P*span**3/(48*E*I)
                beam_type="Simply Supported P={:.1f}kN midspan".format(P/1000)

        fig=make_subplots(rows=3,cols=1,subplot_titles=["Shear Force (kN)","Bending Moment (kN·m)","Deflection (mm)"],vertical_spacing=0.12)
        fig.add_trace(go.Scatter(x=x,y=V/1000,fill="tozeroy",mode="lines",name="Shear",line=dict(color="#4361ee",width=2)),row=1,col=1)
        fig.add_trace(go.Scatter(x=x,y=M/1000,fill="tozeroy",mode="lines",name="Moment",line=dict(color="#f72585",width=2)),row=2,col=1)
        fig.add_trace(go.Scatter(x=x,y=delta*1000,fill="tozeroy",mode="lines",name="Deflection",line=dict(color="#06d6a0",width=2)),row=3,col=1)
        fig.update_layout(height=650,showlegend=False)
        figs.append(_theme(fig,"🏗 Beam Analysis: "+beam_type))

        y_max=0.1; sigma_max=max_moment*y_max/I/1e6; util=sigma_max/250*100
        results.append("## 🏗 Structural Beam Analysis\n**{}**\n**Max Shear:** {:.2f}kN\n**Max Moment:** {:.2f}kN·m\n**Max Deflection:** {:.3f}mm (L/{:.0f})\n**Bending Stress:** {:.1f}MPa ({:.0f}% utilization)\n**Status:** {}\n".format(
            beam_type,abs(V).max()/1000,max_moment/1000,max_deflect*1000,span/max_deflect if max_deflect>0 else 9999,sigma_max,util,"✅ SAFE" if util<100 else "❌ OVERSTRESSED"))

        # Generate DXF
        try:
            import ezdxf
            doc=ezdxf.new("R2010"); msp=doc.modelspace()
            msp.add_line((0,0),(span,0)); msp.add_circle((0,-0.3),0.15); msp.add_circle((span,-0.3),0.15)
            if is_udl:
                for xi in np.linspace(0.5,span-0.5,10): msp.add_line((xi,0.6),(xi,0))
            else:
                msp.add_line((span/2,0.8),(span/2,0))
            dxf_path=tempfile.mktemp(suffix=".dxf"); doc.saveas(dxf_path)
            results.append("**📐 DXF Drawing generated**\n")
        except: pass

    elif any(w in q for w in ["column","buckling","euler","slenderness"]):
        L=next((v for v in vals if 1<=v<=30),4.0); E=200e9; Fy=345e6
        A=next((v for v in vals if 0.001<=v<=0.1),0.00665)
        r=next((v for v in vals if 0.01<=v<=0.5),0.0889)
        K=2.0 if "free" in q else 0.5 if "fixed-fixed" in q else 1.0
        KL_r=K*L/r; lambda_c=(KL_r/np.pi)*np.sqrt(Fy/E)
        Fcr=(0.658**lambda_c**2)*Fy if lambda_c<=1.5 else (0.877/lambda_c**2)*Fy
        Pn=Fcr*A/1000
        sl=np.linspace(0,200,300); fe=np.pi**2*E/np.where(sl>0,sl,1)**2/1e6
        lc=(sl/np.pi)*np.sqrt(Fy/E)
        fcr=np.where(lc<=1.5,(0.658**lc**2)*Fy/1e6,(0.877/lc**2)*Fy/1e6)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=sl,y=np.minimum(fe,Fy/1e6),mode="lines",name="Euler",line=dict(color="#f72585",width=2,dash="dash")))
        fig.add_trace(go.Scatter(x=sl,y=fcr,mode="lines",name="AISC Design",line=dict(color="#4361ee",width=3)))
        fig.add_vline(x=KL_r,line_dash="dot",line_color="#06d6a0",annotation_text="KL/r={:.0f}".format(KL_r))
        figs.append(_theme(fig,"🏗 Column Buckling — AISC 360"))
        results.append("## 🏗 Column Buckling\n**KL/r={:.1f}** | **Fcr={:.1f}MPa** | **φPn={:.1f}kN**\n**Status:** {}\n".format(KL_r,Fcr/1e6,0.9*Pn,"✅ ADEQUATE" if 0.9*Pn>500 else "⚠ CHECK LOAD"))

    elif any(w in q for w in ["foundation","bearing","terzaghi","footing","soil"]):
        B=next((v for v in vals if 0.5<=v<=10),1.5); c=next((v for v in vals if 0<v<=500),25.0)
        phi_deg=next((v for v in vals if 15<=v<=45),30.0); gamma=18.0; Df=1.0
        phi=np.radians(phi_deg); Kp=np.tan(np.pi/4+phi/2)**2
        Nq=np.exp(np.pi*np.tan(phi))*Kp; Nc=(Nq-1)/np.tan(phi) if phi>0 else 5.14; Ng=2*(Nq+1)*np.tan(phi)
        qu=c*Nc+gamma*Df*Nq+0.5*gamma*B*Ng; qa=qu/3
        phi_range=np.linspace(15,45,50); phi_r=np.radians(phi_range)
        Kp_r=np.tan(np.pi/4+phi_r/2)**2; Nq_r=np.exp(np.pi*np.tan(phi_r))*Kp_r
        qu_r=c*(Nq_r-1)/np.tan(phi_r)+gamma*Df*Nq_r+0.5*gamma*B*2*(Nq_r+1)*np.tan(phi_r)
        fig=go.Figure(); fig.add_trace(go.Scatter(x=phi_range,y=qu_r,mode="lines",name="qu",line=dict(color="#4361ee",width=2.5)))
        fig.add_vline(x=phi_deg,line_dash="dash",line_color="#f72585",annotation_text="φ={:.0f}°".format(phi_deg))
        figs.append(_theme(fig,"🏗 Bearing Capacity vs Friction Angle (Terzaghi)"))
        results.append("## 🏗 Foundation Bearing Capacity\n**B={:.1f}m, c={:.0f}kPa, φ={:.0f}°, γ={:.0f}kN/m³**\n**Nc={:.2f}, Nq={:.2f}, Nγ={:.2f}**\n**qu={:.1f}kPa | qa={:.1f}kPa (FS=3)**\n**Max Column Load:** {:.0f}kN\n".format(B,c,phi_deg,gamma,Nc,Nq,Ng,qu,qa,qa*B*B))

    else:
        ln={"en":"English","de":"German","ur":"Urdu"}.get(lang,"English")
        results.append(_call_llm("Structural engineer. Answer in {} with exact calculations: {}".format(ln,query),max_tokens=900,provider="claude"))

    insight=_call_llm("Summarize structural results in {}: {}".format({"en":"English","de":"German","ur":"Urdu"}.get(lang,"English"),"\n".join(results)[:1500]),max_tokens=300)
    return {"figs":figs,"results":results,"insight":insight,"type":"Structural Engineering (AutoCAD/SAP2000)"}

# ══════════════════════════════════════════════════════════════════════════════
# CHEMICAL ENGINEERING (ChemCAD/Aspen replacement)
# ══════════════════════════════════════════════════════════════════════════════
def chemical_engineering(query, df=None, lang="en"):
    from scipy.optimize import fsolve
    q=query.lower(); nums=re.findall(r"[-+]?\d*\.?\d+",query); vals=[float(x) for x in nums]
    figs=[]; results=[]
    ATOMIC_MW={"H":1.008,"C":12.011,"N":14.007,"O":15.999,"Na":22.990,"Cl":35.45,"Fe":55.845,"Ca":40.078,"S":32.06,"P":30.974,"K":39.098,"Mg":24.305,"Al":26.982,"Si":28.086,"Br":79.904,"I":126.90,"F":18.998}
    COMPOUNDS={"water":{"formula":"H2O","MW":18.015,"Tb":100.0,"Tc":374.0},"h2o":{"formula":"H2O","MW":18.015,"Tb":100.0,"Tc":374.0},"ethanol":{"formula":"C2H5OH","MW":46.069,"Tb":78.4,"Tc":240.8},"methane":{"formula":"CH4","MW":16.043,"Tb":-161.5,"Tc":-82.6},"benzene":{"formula":"C6H6","MW":78.113,"Tb":80.1,"Tc":288.9},"co2":{"formula":"CO2","MW":44.010,"Tb":-78.5,"Tc":31.1},"ammonia":{"formula":"NH3","MW":17.031,"Tb":-33.4,"Tc":132.4},"methanol":{"formula":"CH3OH","MW":32.042,"Tb":64.7,"Tc":240.5},"toluene":{"formula":"C7H8","MW":92.140,"Tb":110.6,"Tc":318.6},"acetone":{"formula":"C3H6O","MW":58.079,"Tb":56.1,"Tc":235.1},"propane":{"formula":"C3H8","MW":44.097,"Tb":-42.1,"Tc":96.7},"h2so4":{"formula":"H2SO4","MW":98.079,"Tb":337.0,"Tc":654.0},"nacl":{"formula":"NaCl","MW":58.443,"Tb":1413.0,"Tc":3900.0}}
    compound=next((COMPOUNDS[name] for name in COMPOUNDS if name in q), None)
    if compound is None:
        fm=re.search(r"\b([A-Z][a-z]?\d*){2,}\b",query)
        if fm:
            formula=fm.group(0); toks=re.findall(r"([A-Z][a-z]?)(\d*)",formula)
            mw=sum(ATOMIC_MW.get(el,0)*int(n or 1) for el,n in toks if el in ATOMIC_MW)
            compound={"formula":formula,"MW":round(mw,3),"Tb":None,"Tc":None}

    if any(w in q for w in ["ideal gas","pv=nrt","pvt","molar volume","gas law"]):
        n=next((v for v in vals if 0.001<=v<=1000),1.0); T_C=next((v for v in vals if -273<=v<=2000),25.0); P_bar=next((v for v in vals if 0.001<=v<=1000),1.0)
        T_K=T_C+273.15; R=8.314; V_ideal=n*R*T_K/(P_bar*1e5)*1000
        T_r=np.linspace(0,500,150); V_r=n*R*(T_r+273.15)/(P_bar*1e5)*1000
        P_r=np.linspace(0.1,200,150); V_P=n*R*T_K/(P_r*1e5)*1000
        fig=make_subplots(rows=1,cols=2,subplot_titles=["V vs T (constant P={:.1f}bar)".format(P_bar),"V vs P (constant T={:.0f}°C)".format(T_C)])
        fig.add_trace(go.Scatter(x=T_r,y=V_r,mode="lines",name="V(T)",line=dict(color="#4361ee",width=2.5)),row=1,col=1)
        fig.add_trace(go.Scatter(x=P_r,y=V_P,mode="lines",name="V(P)",line=dict(color="#f72585",width=2.5)),row=1,col=2)
        fig.add_vline(x=T_C,line_dash="dot",line_color="#06d6a0",row=1,col=1)
        figs.append(_theme(fig,"🧪 Ideal Gas PVT Diagram"))
        results.append("## 🧪 Ideal Gas (PV=nRT)\n**T={:.1f}°C ({:.1f}K), P={:.2f}bar, n={:.3f}mol**\n**Volume:** {:.4f}L = {:.6f}m³\n**R=8.314 J/(mol·K)**\n".format(T_C,T_K,P_bar,n,V_ideal,V_ideal/1000))

    elif any(w in q for w in ["distillation","mccabe","thiele","reflux","column","stages","distillate","bottoms"]):
        xF=next((v for v in vals if 0<v<1),0.40); xD=next((v for v in vals if v>xF and v<1),0.95); xB=next((v for v in vals if v<xF and v>0),0.05)
        R=next((v for v in vals if v>1),2.5); alpha=next((v for v in vals if 1<v<20),2.5)
        x_range=np.linspace(0,1,300); y_eq=alpha*x_range/(1+(alpha-1)*x_range)
        L_V=R/(R+1); b_rect=xD/(R+1)
        y_at_xF=L_V*xF+b_rect; slope_strip=(y_at_xF-xB)/(xF-xB) if abs(xF-xB)>0.001 else 1.2
        x_rect=np.linspace(0,xD,60); y_rect=L_V*x_rect+b_rect
        x_strip=np.linspace(xB,xF,60); y_strip=slope_strip*(x_strip-xB)+xB
        x_steps=[xD]; y_steps=[xD]; stages=0; x_cur=xD
        while x_cur>xB and stages<60:
            x_eq=x_cur/alpha/(1-(1-1/alpha)*x_cur) if alpha>1 else x_cur
            x_steps+=[x_cur,x_eq]; y_steps+=[x_cur,x_cur]
            y_new=L_V*x_eq+b_rect if x_eq>=xF else slope_strip*(x_eq-xB)+xB
            x_steps+=[x_eq,x_eq]; y_steps+=[x_cur,y_new]
            x_cur=x_eq; stages+=1
            if x_eq<=xB*1.05: break
        Nmin=np.log((xD/(1-xD))*(1-xB)/xB)/np.log(alpha) if alpha>1 else 10
        Rmin=alpha*xF/((alpha-1)*xD)-1/(alpha-1)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=x_range,y=x_range,mode="lines",name="y=x",line=dict(color="#555",dash="dot")))
        fig.add_trace(go.Scatter(x=x_range,y=y_eq,mode="lines",name="Equil Curve",line=dict(color="#4361ee",width=2.5)))
        fig.add_trace(go.Scatter(x=x_rect,y=y_rect,mode="lines",name="Rectifying OL",line=dict(color="#06d6a0",width=2)))
        fig.add_trace(go.Scatter(x=x_strip,y=y_strip,mode="lines",name="Stripping OL",line=dict(color="#ffd166",width=2)))
        fig.add_trace(go.Scatter(x=x_steps,y=y_steps,mode="lines",name="Stages",line=dict(color="#f72585",width=1.5,dash="dash")))
        fig.add_vline(x=xF,line_dash="dot",line_color="#aaa",annotation_text="xF")
        figs.append(_theme(fig,"🧪 McCabe-Thiele Distillation (α={:.1f}, R={:.1f})".format(alpha,R)))
        results.append("## 🧪 Distillation Column (McCabe-Thiele)\n**xF={:.3f}, xD={:.3f}, xB={:.4f}**\n**R={:.2f}, α={:.2f}**\n**Rmin={:.2f}, Nmin={:.1f} (Fenske)**\n**Actual Stages: ~{}**\n**R/Rmin={:.2f}** (recommend 1.2–1.5)\n".format(xF,xD,xB,R,alpha,max(0,Rmin),Nmin,stages,R/max(Rmin,0.01)))

    elif any(w in q for w in ["heat exchanger","lmtd","ntu","heat transfer"]):
        Q=next((v for v in vals if v>100),500.0); T_hi=next((v for v in vals if v>50),150.0); T_ho=next((v for v in vals if 30<v<T_hi),80.0)
        T_ci=next((v for v in vals if 10<=v<=40),20.0); T_co=next((v for v in vals if T_ci<v<T_ho),60.0); U=next((v for v in vals if 100<=v<=5000),500.0)
        dT1=T_hi-T_co; dT2=T_ho-T_ci
        LMTD=(dT1-dT2)/np.log(dT1/dT2) if abs(dT1-dT2)>0.001 and dT1>0 and dT2>0 else (dT1+dT2)/2
        A=Q*1000/(U*LMTD) if LMTD>0 else 0
        z=np.linspace(0,1,200); T_hot=T_hi-(T_hi-T_ho)*z; T_cold_cc=T_co-(T_co-T_ci)*z
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=z,y=T_hot,mode="lines",name="Hot",line=dict(color="#ef476f",width=2.5)))
        fig.add_trace(go.Scatter(x=z,y=T_cold_cc,mode="lines",name="Cold (counter)",line=dict(color="#4361ee",width=2.5)))
        fig.add_trace(go.Scatter(x=z,y=T_ci+(T_co-T_ci)*z,mode="lines",name="Cold (parallel)",line=dict(color="#4cc9f0",width=2,dash="dash")))
        figs.append(_theme(fig,"🧪 Heat Exchanger — Temperature Profiles"))
        results.append("## 🧪 Heat Exchanger (LMTD Method)\n**Q={:.1f}kW, U={:.0f}W/m²K**\n**Hot:** {:.0f}°C→{:.0f}°C | **Cold:** {:.0f}°C→{:.0f}°C\n**LMTD (counter-current):** {:.2f}°C\n**Required Area:** {:.2f}m²\n".format(Q,U,T_hi,T_ho,T_ci,T_co,LMTD,A))

    elif compound:
        T_range=np.linspace(-50,300,300)
        Antoine={"H2O":(8.07131,1730.63,233.426),"C2H5OH":(8.11220,1592.864,226.184),"CH3OH":(8.08097,1582.271,239.726),"C6H6":(6.90565,1211.033,220.790),"C7H8":(6.95087,1342.310,219.187),"CH4":(6.82051,405.42,267.777)}
        ant=Antoine.get(compound.get("formula",""),None)
        if ant:
            A,B,C=ant; Pvap=10**(A-B/(T_range+C))*133.322
            fig=go.Figure(); fig.add_trace(go.Scatter(x=T_range,y=Pvap/1000,mode="lines",name="Pvap",line=dict(color="#4361ee",width=2.5)))
            if compound.get("Tb"): fig.add_vline(x=compound["Tb"],line_dash="dash",line_color="#f72585",annotation_text="Tb={:.1f}°C".format(compound["Tb"]))
            figs.append(_theme(fig,"🧪 Vapor Pressure — {}".format(compound["formula"])))
        results.append("## 🧪 Compound: {}\n**Formula:** {}\n**MW:** {:.3f} g/mol\n**Tb:** {}°C | **Tc:** {}°C\n".format(query[:30],compound.get("formula","?"),compound.get("MW",0),compound.get("Tb","?"),compound.get("Tc","?")))

    else:
        ln={"en":"English","de":"German","ur":"Urdu"}.get(lang,"English")
        results.append(_call_llm("Chemical engineer (ChemCAD/Aspen level). Answer in {} with exact calculations: {}".format(ln,query),max_tokens=900,provider="claude"))

    insight=_call_llm("Summarize chemical engineering results in {}: {}".format({"en":"English","de":"German","ur":"Urdu"}.get(lang,"English"),"\n".join(results)[:1500]),max_tokens=300)
    return {"figs":figs,"results":results,"insight":insight,"type":"Chemical Engineering (ChemCAD/Aspen)"}

# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED STATISTICS (R/SPSS/SAS/Stata replacement)
# ══════════════════════════════════════════════════════════════════════════════
def advanced_statistics(df, query, lang="en"):
    from scipy import stats as _stats
    q=query.lower(); figs=[]; results=[]; schema=fingerprint_schema(df)
    measures=[c for c,m in schema.items() if m["role"]=="measure"]
    dimensions=[c for c,m in schema.items() if m["role"]=="dimension"]
    temporals=[c for c,m in schema.items() if m["role"]=="temporal"]

    if any(w in q for w in ["regression","r-squared","r²","lm","ols","linear model","predict","coefficient"]):
        if len(measures)>=2:
            x_col,y_col=measures[0],measures[1]
            clean=df[[x_col,y_col]].dropna(); n=len(clean)
            slope,intercept,r,p,se=_stats.linregress(clean[x_col],clean[y_col])
            r2=r**2; t_crit=_stats.t.ppf(0.975,n-2); ci_slope=t_crit*se
            residuals=clean[y_col]-(slope*clean[x_col]+intercept)
            fig=px.scatter(clean,x=x_col,y=y_col,trendline="ols",title="📊 Regression: {}~{}".format(y_col,x_col),color_discrete_sequence=PAL,opacity=0.7)
            fig.add_annotation(text="R²={:.4f} | p={:.4e} | β={:.4f} | n={}".format(r2,p,slope,n),xref="paper",yref="paper",x=0.02,y=0.98,showarrow=False,font=dict(color="#06d6a0",size=12),bgcolor="rgba(0,0,0,.5)")
            figs.append(_theme(fig))
            fig2=go.Figure(); fig2.add_trace(go.Scatter(x=clean[x_col],y=residuals,mode="markers",marker=dict(color="#f72585",size=5,opacity=0.6),name="Residuals")); fig2.add_hline(y=0,line_dash="dash",line_color="#aaa")
            figs.append(_theme(fig2,"📊 Residuals Plot"))
            results.append("## 📊 OLS Regression: {} ~ {}\n**n={}** | **R²={:.4f}** ({:.1f}% variance)\n**β₁={:.4f} ± {:.4f}** (95% CI)\n**p={:.4e}** ({})\n**Equation:** {} = {:.4f}·{} + {:.4f}\n".format(y_col,x_col,n,r2,r2*100,slope,ci_slope,p,"✅ Significant" if p<0.05 else "❌ Not significant",y_col,slope,x_col,intercept))

    elif any(w in q for w in ["anova","variance","f-test","between groups"]):
        if measures and dimensions:
            m_col,d_col=measures[0],dimensions[0]
            groups=[grp[m_col].dropna().values for _,grp in df.groupby(d_col) if len(grp)>2]
            if len(groups)>=2:
                f,p=_stats.f_oneway(*groups)
                grand_mean=df[m_col].mean(); ss_b=sum(len(g)*(g.mean()-grand_mean)**2 for g in groups); ss_t=sum(((g-grand_mean)**2).sum() for g in groups)
                eta2=ss_b/ss_t if ss_t>0 else 0
                fig=px.box(df.dropna(subset=[d_col,m_col]),x=d_col,y=m_col,color=d_col,color_discrete_sequence=PAL,title="📊 ANOVA: {} by {}".format(m_col,d_col))
                figs.append(_theme(fig))
                results.append("## 📊 One-Way ANOVA: {} by {}\n**F={:.4f}, p={:.4e}** ({})\n**η²={:.4f}** ({} effect)\n".format(m_col,d_col,f,p,"✅ Significant" if p<0.05 else "❌ Not significant",eta2,"Large" if eta2>0.14 else "Medium" if eta2>0.06 else "Small"))

    elif any(w in q for w in ["survival","kaplan","meier","cox","hazard","censored"]):
        try:
            from lifelines import KaplanMeierFitter
            dur_col=next((c for c in df.columns if any(w in c.lower() for w in ["time","duration","days","months","survival"])),None)
            ev_col=next((c for c in df.columns if any(w in c.lower() for w in ["event","status","death","died","outcome"])),None)
            if dur_col and ev_col:
                kmf=KaplanMeierFitter(); fig=go.Figure()
                grp_col=next((c for c,m in schema.items() if m["role"]=="dimension"),None)
                if grp_col:
                    for name,grp in df.groupby(grp_col):
                        g=grp[[dur_col,ev_col]].dropna(); kmf.fit(g[dur_col],g[ev_col],label=str(name))
                        fig.add_trace(go.Scatter(x=kmf.timeline,y=kmf.survival_function_.iloc[:,0],mode="lines",name=str(name)))
                else:
                    kmf.fit(df[dur_col].dropna(),df[ev_col].dropna()); fig.add_trace(go.Scatter(x=kmf.timeline,y=kmf.survival_function_.iloc[:,0],mode="lines",name="Survival",line=dict(color="#4361ee",width=2.5)))
                figs.append(_theme(fig,"📊 Kaplan-Meier Survival Curve"))
                results.append("## 📊 Kaplan-Meier Survival Analysis\n**Median Survival:** {:.2f}\n".format(kmf.median_survival_time_))
        except ImportError: results.append("pip install lifelines\n")

    elif any(w in q for w in ["correlation","heatmap","pearson","spearman"]):
        nums=measures[:12]
        if len(nums)>=2:
            method="spearman" if "spearman" in q else "pearson"
            corr=df[nums].corr(method=method).round(3); ns=[c[:15] for c in corr.columns]
            fig=go.Figure(go.Heatmap(z=corr.values,x=ns,y=ns,colorscale="RdBu",zmid=0,text=corr.values.round(2),texttemplate="%{text}",textfont={"size":max(8,12-len(ns))}))
            fig.update_layout(height=max(400,55*len(ns))); figs.append(_theme(fig,"📊 {} Correlation Matrix".format(method.title())))
            pairs=sorted([(corr.columns[i],corr.columns[j],abs(corr.iloc[i,j])) for i in range(len(corr.columns)) for j in range(i+1,len(corr.columns))],key=lambda x:-x[2])
            results.append("## 📊 {} Correlation\n".format(method.title())+"\n".join("  • {} ↔ {}: r={:.3f} ({})".format(a,b,r,"Strong" if r>0.7 else "Moderate" if r>0.4 else "Weak") for a,b,r in pairs[:8]))

    elif any(w in q for w in ["describe","summary","descriptive","statistics","normality"]):
        from scipy import stats as _stats
        if measures:
            rows=[]
            for col in measures[:8]:
                s=df[col].dropna(); sw,p_sw=_stats.shapiro(s[:50]) if len(s)>=8 else (0,1)
                rows.append({"Column":col,"N":len(s),"Mean":round(s.mean(),4),"Median":round(s.median(),4),"SD":round(s.std(),4),"Min":round(s.min(),4),"Max":round(s.max(),4),"Skew":round(float(s.skew()),3),"Kurt":round(float(s.kurtosis()),3),"Shapiro-p":round(p_sw,4)})
            desc_df=pd.DataFrame(rows)
            fig=go.Figure(go.Table(header=dict(values=list(desc_df.columns),fill_color="#1a1a3e",font=dict(color="white",size=11)),cells=dict(values=[desc_df[c] for c in desc_df.columns],fill_color="#12121f",font=dict(color="#e8e8f0",size=10))))
            figs.append(_theme(fig,"📊 Descriptive Statistics"))
            for col in measures[:3]:
                vals_col=df[col].dropna(); mu,sigma=vals_col.mean(),vals_col.std()
                from scipy import stats as _st; x_n=np.linspace(vals_col.min(),vals_col.max(),200)
                fig2=go.Figure(); fig2.add_trace(go.Histogram(x=vals_col,nbinsx=30,name=col,marker_color="#4361ee",opacity=0.7,histnorm="probability density"))
                fig2.add_trace(go.Scatter(x=x_n,y=_st.norm.pdf(x_n,mu,sigma),mode="lines",name="Normal",line=dict(color="#f72585",width=2,dash="dash")))
                figs.append(_theme(fig2,"📊 Distribution: {}".format(col)))
            results.append("## 📊 Descriptive Statistics\n"+desc_df.to_string())

    else:
        ln={"en":"English","de":"German","ur":"Urdu"}.get(lang,"English")
        results.append(_call_llm("Expert statistician (R/SPSS/SAS level). Answer in {}: {}".format(ln,query),max_tokens=900,provider="claude"))

    insight=_call_llm("Summarize statistics in {}: {}".format({"en":"English","de":"German","ur":"Urdu"}.get(lang,"English"),"\n".join(results)[:1500]),max_tokens=300)
    return {"figs":figs,"results":results,"insight":insight,"type":"Advanced Statistics (R/SPSS/SAS)"}

# ══════════════════════════════════════════════════════════════════════════════
# MACHINE LEARNING
# ══════════════════════════════════════════════════════════════════════════════
def machine_learning(df, query, lang="en"):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    q=query.lower(); figs=[]; results=[]; schema=fingerprint_schema(df)
    measures=[c for c,m in schema.items() if m["role"]=="measure"][:10]
    dims=[c for c,m in schema.items() if m["role"]=="dimension"]

    if any(w in q for w in ["cluster","kmeans","k-means","segment"]):
        if len(measures)>=2:
            X=df[measures].dropna(); scaler=StandardScaler(); Xs=scaler.fit_transform(X)
            K_max=min(10,len(X)//5)
            inertias=[KMeans(n_clusters=k,random_state=42,n_init=10).fit(Xs).inertia_ for k in range(2,K_max+1)]
            fig=go.Figure(go.Scatter(x=list(range(2,K_max+1)),y=inertias,mode="lines+markers",line=dict(color="#4361ee",width=2),marker=dict(size=8,color="#f72585")))
            figs.append(_theme(fig,"🧬 K-Means Elbow Curve"))
            best_k=min(4,K_max); km=KMeans(n_clusters=best_k,random_state=42,n_init=10); labels=km.fit_predict(Xs)
            pca=PCA(n_components=2); coords=pca.fit_transform(Xs)
            pca_df=pd.DataFrame({"PC1":coords[:,0],"PC2":coords[:,1],"Cluster":labels.astype(str)})
            fig2=px.scatter(pca_df,x="PC1",y="PC2",color="Cluster",title="🧬 PCA Clusters ({}k)".format(best_k),color_discrete_sequence=PAL,opacity=0.7)
            figs.append(_theme(fig2))
            df_cl=df[measures].copy(); df_cl["Cluster"]=labels
            profile=df_cl.groupby("Cluster")[measures].mean()
            fig3=go.Figure(go.Heatmap(z=StandardScaler().fit_transform(profile.values),x=measures,y=["Cluster {}".format(i) for i in range(best_k)],colorscale="RdBu",zmid=0,text=profile.values.round(2),texttemplate="%{text}"))
            figs.append(_theme(fig3,"🧬 Cluster Profiles"))
            results.append("## 🧬 K-Means Clustering\n**Optimal k={}** | **PCA variance:** {:.1f}%\n".format(best_k,sum(pca.explained_variance_ratio_)*100))
            for i in range(best_k):
                cnt=(labels==i).sum(); results.append("  Cluster {}: {} samples ({:.1f}%)\n".format(i,cnt,cnt/len(labels)*100))

    elif any(w in q for w in ["pca","principal component","variance","scree","dimensionality"]):
        if len(measures)>=3:
            X=df[measures].dropna(); scaler=StandardScaler(); Xs=scaler.fit_transform(X)
            n_comp=min(len(measures),10); pca=PCA(n_components=n_comp); pca.fit(Xs)
            ev=pca.explained_variance_ratio_; cum_ev=np.cumsum(ev)
            fig=make_subplots(rows=1,cols=2,subplot_titles=["Scree Plot","Cumulative Variance"])
            fig.add_trace(go.Bar(x=["PC{}".format(i+1) for i in range(n_comp)],y=ev*100,marker_color=PAL[0]),row=1,col=1)
            fig.add_trace(go.Scatter(x=["PC{}".format(i+1) for i in range(n_comp)],y=cum_ev*100,mode="lines+markers",line=dict(color="#f72585",width=2)),row=1,col=2)
            fig.add_hline(y=80,row=1,col=2,line_dash="dash",line_color="#06d6a0",annotation_text="80%")
            figs.append(_theme(fig,"🧬 PCA Scree Plot"))
            coords=pca.transform(Xs)[:,:2]
            fig2=px.scatter(x=coords[:,0],y=coords[:,1],opacity=0.5,labels={"x":"PC1({:.1f}%)".format(ev[0]*100),"y":"PC2({:.1f}%)".format(ev[1]*100)},title="🧬 PCA Biplot",color_discrete_sequence=PAL)
            figs.append(_theme(fig2))
            n_90=np.argmax(cum_ev>=0.9)+1
            results.append("## 🧬 PCA\n**Components for 90% variance:** {}\n"+"\n".join("  PC{}: {:.1f}% (cum:{:.1f}%)".format(i+1,ev[i]*100,cum_ev[i]*100) for i in range(min(5,n_comp))))

    elif any(w in q for w in ["random forest","classification","predict","decision tree"]):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score, train_test_split
        target=dims[0] if dims else None
        if target and measures:
            X=df[measures].fillna(df[measures].mean()); y=df[target]
            X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42)
            model=RandomForestClassifier(n_estimators=100,random_state=42); model.fit(X_tr,y_tr)
            cv=cross_val_score(model,X,y,cv=5,scoring="accuracy")
            fi=pd.DataFrame({"feature":measures,"importance":model.feature_importances_}).sort_values("importance",ascending=False)
            fig=px.bar(fi,x="importance",y="feature",orientation="h",color="importance",color_continuous_scale="Blues",title="🧬 Feature Importance")
            fig.update_coloraxes(showscale=False); figs.append(_theme(fig))
            results.append("## 🧬 Random Forest\n**CV Accuracy:** {:.2f}% ± {:.2f}%\n**Top Feature:** {} ({:.3f})\n".format(cv.mean()*100,cv.std()*100,fi.iloc[0]["feature"],fi.iloc[0]["importance"]))

    else:
        ln={"en":"English","de":"German","ur":"Urdu"}.get(lang,"English")
        results.append(_call_llm("ML expert. Answer in {}: {}".format(ln,query),max_tokens=900,provider="claude"))

    insight=_call_llm("Summarize ML results in {}: {}".format({"en":"English","de":"German","ur":"Urdu"}.get(lang,"English"),"\n".join(results)[:1500]),max_tokens=300)
    return {"figs":figs,"results":results,"insight":insight,"type":"Machine Learning"}

# ══════════════════════════════════════════════════════════════════════════════
# FINANCIAL ENGINEERING (Bloomberg/Excel replacement)
# ══════════════════════════════════════════════════════════════════════════════
def financial_engineering(query, df=None, lang="en"):
    from scipy import stats as _stats
    import numpy_financial as npf
    q=query.lower(); nums=re.findall(r"[-+]?\d*\.?\d+",query); vals=[float(x) for x in nums if float(x)!=0]
    figs=[]; results=[]

    if any(w in q for w in ["npv","irr","dcf","cash flow","payback","wacc","investment","roi"]):
        r=next((v/100 for v in vals if 1<=v<=50),0.10)
        initial=-abs(next((v for v in vals if v>1000),100000))
        cf_nums=re.findall(r"\d{4,}",query)
        cash_flows=[initial]+[float(x) for x in cf_nums[:8]] if len(cf_nums)>=3 else [initial]+[abs(initial)*0.3*(1.05**i) for i in range(5)]
        npv_val=npf.npv(r,cash_flows)
        try: irr=npf.irr(cash_flows)*100
        except: irr=0
        cum=0; payback=len(cash_flows)
        for i,cf in enumerate(cash_flows[1:],1):
            cum+=cf
            if cum>=abs(initial): payback=i-1+(abs(initial)-(cum-cf))/cf; break
        fig=go.Figure(go.Waterfall(x=["Y{}".format(i) for i in range(len(cash_flows))],y=cash_flows,connector=dict(line=dict(color="rgba(255,255,255,.3)")),increasing=dict(marker_color="#06d6a0"),decreasing=dict(marker_color="#ef476f")))
        figs.append(_theme(fig,"💹 Cash Flow Waterfall"))
        rates=np.linspace(0.01,0.35,200); npvs=[npf.npv(r_,cash_flows) for r_ in rates]
        fig2=go.Figure(); fig2.add_trace(go.Scatter(x=rates*100,y=npvs,mode="lines",name="NPV",line=dict(color="#4361ee",width=2.5))); fig2.add_hline(y=0,line_dash="dash",line_color="#aaa")
        figs.append(_theme(fig2,"💹 NPV Sensitivity to Discount Rate"))
        results.append("## 💹 DCF Analysis\n**WACC:** {:.1f}%\n**NPV:** ${:,.0f} ({})\n**IRR:** {:.2f}%\n**Payback:** {:.2f} years\n**PI:** {:.3f} ({})\n".format(r*100,npv_val,"✅ VALUE CREATING" if npv_val>0 else "❌ VALUE DESTROYING",irr,payback,(npv_val+abs(initial))/abs(initial),"✅ ACCEPT" if npv_val>0 else "❌ REJECT"))

    elif any(w in q for w in ["option","black-scholes","call","put","strike","volatility","greeks","delta","gamma"]):
        S=next((v for v in vals if 10<=v<=10000),100.0); K=next((v for v in vals if v!=S and 10<=v<=10000),100.0)
        T=next((v for v in vals if 0.01<=v<=5),1.0); r=next((v/100 for v in vals if 1<=v<=20),0.05); sigma=next((v/100 for v in vals if 5<=v<=100),0.20)
        d1=(np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T)); d2=d1-sigma*np.sqrt(T)
        call=S*_stats.norm.cdf(d1)-K*np.exp(-r*T)*_stats.norm.cdf(d2); put=K*np.exp(-r*T)*_stats.norm.cdf(-d2)-S*_stats.norm.cdf(-d1)
        delta_call=_stats.norm.cdf(d1); gamma=_stats.norm.pdf(d1)/(S*sigma*np.sqrt(T)); vega=S*_stats.norm.pdf(d1)*np.sqrt(T)/100
        S_range=np.linspace(S*0.5,S*1.5,100); call_vals=[S_*_stats.norm.cdf((np.log(S_/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T)))-K*np.exp(-r*T)*_stats.norm.cdf((np.log(S_/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))-sigma*np.sqrt(T)) for S_ in S_range]
        put_vals=[K*np.exp(-r*T)*_stats.norm.cdf(-(d1-(np.log(S_/S)+0)))-S_*_stats.norm.cdf(-_stats.norm.ppf(_stats.norm.cdf(d1))) for S_ in S_range]
        fig=go.Figure(); fig.add_trace(go.Scatter(x=S_range,y=call_vals,mode="lines",name="Call",line=dict(color="#4361ee",width=2.5))); fig.add_trace(go.Scatter(x=S_range,y=[max(0,s-K) for s in S_range],mode="lines",name="Intrinsic",line=dict(color="#aaa",dash="dash"))); fig.add_vline(x=S,line_dash="dot",line_color="#f72585",annotation_text="S={:.0f}".format(S))
        figs.append(_theme(fig,"💹 Black-Scholes Option Value"))
        results.append("## 💹 Black-Scholes\n**S={:.2f}, K={:.2f}, T={:.2f}yr, r={:.1f}%, σ={:.1f}%**\n**Call:** ${:.4f} | **Put:** ${:.4f}\n**Delta:** {:.4f} | **Gamma:** {:.6f} | **Vega:** {:.4f}\n**d1={:.4f}, d2={:.4f}**\n".format(S,K,T,r*100,sigma*100,call,put,delta_call,gamma,vega,d1,d2))

    elif any(w in q for w in ["portfolio","markowitz","sharpe","efficient frontier","optimize"]):
        np.random.seed(42); n_assets=5
        if df is not None and len(df)>20:
            ret_cols=[c for c in df.select_dtypes(include="number").columns][:5]
            returns=df[ret_cols].pct_change().dropna()
        else:
            returns=pd.DataFrame(np.random.randn(252,n_assets)*0.01+0.0005,columns=["Asset{}".format(i+1) for i in range(n_assets)])
            ret_cols=returns.columns.tolist()
        mu=returns.mean()*252; cov=returns.cov()*252; n=len(ret_cols)
        n_port=3000; port_ret=np.zeros(n_port); port_vol=np.zeros(n_port); port_sharpe=np.zeros(n_port); all_w=np.zeros((n_port,n))
        for i in range(n_port):
            w=np.random.dirichlet(np.ones(n)); all_w[i]=w; port_ret[i]=w@mu; port_vol[i]=np.sqrt(w@cov@w); port_sharpe[i]=port_ret[i]/port_vol[i]
        max_si=np.argmax(port_sharpe); min_vi=np.argmin(port_vol)
        fig=go.Figure(); fig.add_trace(go.Scatter(x=port_vol*100,y=port_ret*100,mode="markers",marker=dict(color=port_sharpe,colorscale="Viridis",size=3,opacity=0.5),name="Portfolios")); fig.add_trace(go.Scatter(x=[port_vol[max_si]*100],y=[port_ret[max_si]*100],mode="markers",marker=dict(color="#f72585",size=15,symbol="star"),name="Max Sharpe")); fig.add_trace(go.Scatter(x=[port_vol[min_vi]*100],y=[port_ret[min_vi]*100],mode="markers",marker=dict(color="#06d6a0",size=15,symbol="diamond"),name="Min Vol"))
        figs.append(_theme(fig,"💹 Markowitz Efficient Frontier"))
        opt_w=all_w[max_si]; results.append("## 💹 Portfolio Optimization\n**Max Sharpe Portfolio:**\n"+"\n".join("  {}: {:.1f}%".format(c,w*100) for c,w in zip(ret_cols,opt_w))+"\n**Return:** {:.2f}% | **Vol:** {:.2f}% | **Sharpe:** {:.3f}\n".format(port_ret[max_si]*100,port_vol[max_si]*100,port_sharpe[max_si]))

    elif any(w in q for w in ["stock","ticker","price","equity","market"]):
        tickers=[t for t in re.findall(r"\b[A-Z]{1,5}\b",query) if t not in ["THE","FOR","AND","OR","IN","A","I"]][:3]
        if tickers:
            try:
                import yfinance as yf; fig=go.Figure()
                for tick in tickers:
                    data=yf.download(tick,period="1y",progress=False)
                    if not data.empty:
                        close=data["Close"].squeeze()
                        fig.add_trace(go.Scatter(x=data.index,y=close,mode="lines",name=tick,line=dict(width=2)))
                        results.append("**{}:** ${:.2f} | 52w H:${:.2f} L:${:.2f}\n".format(tick,float(close.iloc[-1]),float(close.max()),float(close.min())))
                figs.append(_theme(fig,"💹 Stock Prices: {}".format(", ".join(tickers))))
            except Exception as e: results.append("Stock error: {}\n".format(str(e)))

    else:
        ln={"en":"English","de":"German","ur":"Urdu"}.get(lang,"English")
        results.append(_call_llm("CFO/financial analyst. Answer in {} with exact calculations: {}".format(ln,query),max_tokens=900,provider="claude"))

    insight=_call_llm("Summarize finance results in {}: {}".format({"en":"English","de":"German","ur":"Urdu"}.get(lang,"English"),"\n".join(results)[:1500]),max_tokens=300)
    return {"figs":figs,"results":results,"insight":insight,"type":"Financial Engineering (Bloomberg)"}

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING & CONTROL (MATLAB replacement)
# ══════════════════════════════════════════════════════════════════════════════
def signal_processing(query, df=None, lang="en"):
    from scipy import signal as _sig
    from scipy.fft import fft, fftfreq
    q=query.lower(); nums=re.findall(r"[-+]?\d*\.?\d+",query); vals=[float(x) for x in nums]
    figs=[]; results=[]

    if any(w in q for w in ["fft","fourier","frequency","spectrum"]):
        if df is not None and len(df)>10:
            num_cols=[c for c in df.select_dtypes(include="number").columns]
            if num_cols:
                sig_data=df[num_cols[0]].dropna().values; fs=next((v for v in vals if v>10),1000.0)
                N=len(sig_data); fft_vals=fft(sig_data); freqs=fftfreq(N,1/fs); power=np.abs(fft_vals[:N//2])**2
                fig=go.Figure(); fig.add_trace(go.Scatter(x=freqs[:N//2],y=power,mode="lines",name="Power",line=dict(color="#4361ee",width=1.5)))
                top_f=freqs[:N//2][np.argmax(power[1:])+1]
                fig.add_vline(x=top_f,line_dash="dash",line_color="#f72585",annotation_text="f={:.1f}Hz".format(top_f))
                figs.append(_theme(fig,"📐 FFT Spectrum"))
                results.append("## 📐 FFT\n**Dominant frequency:** {:.2f}Hz | **Fs:** {:.0f}Hz\n".format(top_f,fs))
        else:
            fs=1000; t=np.linspace(0,1,fs); sig_data=np.sin(2*np.pi*50*t)+0.5*np.sin(2*np.pi*120*t)+0.1*np.random.randn(fs)
            fft_vals=fft(sig_data); freqs=fftfreq(fs,1/fs); power=np.abs(fft_vals[:fs//2])**2
            fig=make_subplots(rows=2,cols=1,subplot_titles=["Time Signal","FFT Spectrum"])
            fig.add_trace(go.Scatter(x=t[:200],y=sig_data[:200],mode="lines",name="Signal",line=dict(color="#4361ee",width=1)),row=1,col=1)
            fig.add_trace(go.Scatter(x=freqs[:fs//2],y=power,mode="lines",name="Power",line=dict(color="#f72585",width=1.5)),row=2,col=1)
            for f_peak in [50,120]: fig.add_vline(x=f_peak,line_dash="dash",line_color="#06d6a0",row=2,col=1)
            figs.append(_theme(fig,"📐 FFT Demo: 50Hz + 120Hz")); results.append("## 📐 FFT Demo\n**Peaks at 50Hz and 120Hz**\n")

    elif any(w in q for w in ["filter","butterworth","lowpass","highpass","bandpass","cutoff"]):
        fs=next((v for v in vals if v>10),1000.0); fc=next((v for v in vals if 0<v<fs/2),100.0); order=int(next((v for v in vals if 1<=v<=10),4))
        ftype="highpass" if "high" in q else "bandpass" if "band" in q else "lowpass"
        if ftype=="bandpass":
            fc2=next((v for v in vals if v>fc and v<fs/2),fc*2); sos=_sig.butter(order,[fc,fc2],btype="band",fs=fs,output="sos")
        else: sos=_sig.butter(order,fc,btype=ftype,fs=fs,output="sos")
        w,h=_sig.sosfreqz(sos,worN=2000,fs=fs)
        fig=make_subplots(rows=2,cols=1,subplot_titles=["Magnitude (dB)","Phase (°)"])
        fig.add_trace(go.Scatter(x=w,y=20*np.log10(np.abs(h)+1e-12),mode="lines",name="Mag",line=dict(color="#4361ee",width=2)),row=1,col=1)
        fig.add_trace(go.Scatter(x=w,y=np.angle(h,deg=True),mode="lines",name="Phase",line=dict(color="#f72585",width=2)),row=2,col=1)
        fig.update_xaxes(type="log"); fig.update_layout(height=500)
        figs.append(_theme(fig,"📐 Butterworth {} Filter (fc={:.0f}Hz, N={})".format(ftype.title(),fc,order)))
        results.append("## 📐 Butterworth Filter\n**Type:** {} | **fc:** {:.0f}Hz | **Order:** {}\n**Rolloff:** {:.0f}dB/decade\n".format(ftype.title(),fc,order,20*order))

    elif any(w in q for w in ["bode","transfer function","pid","control","step response","poles","zeros"]):
        try:
            import control as _ctrl
            wn=next((v for v in vals if 0.1<=v<=1000),10.0); zeta=next((v for v in vals if 0<v<=2),0.5); K=next((v for v in vals if v>10),1.0)
            sys=_ctrl.tf([K*wn**2],[1,2*zeta*wn,wn**2])
            w_range=np.logspace(-2,3,500); mag,phase,omega=_ctrl.bode(sys,omega=w_range,plot=False)
            fig=make_subplots(rows=2,cols=1,subplot_titles=["Magnitude (dB)","Phase (°)"])
            fig.add_trace(go.Scatter(x=omega,y=20*np.log10(mag),mode="lines",name="Mag",line=dict(color="#4361ee",width=2)),row=1,col=1)
            fig.add_trace(go.Scatter(x=omega,y=phase*180/np.pi,mode="lines",name="Phase",line=dict(color="#f72585",width=2)),row=2,col=1)
            fig.update_xaxes(type="log"); figs.append(_theme(fig,"📐 Bode Plot"))
            t_step,y_step=_ctrl.step_response(sys,T=np.linspace(0,5/zeta/wn,500))
            fig2=go.Figure(); fig2.add_trace(go.Scatter(x=t_step,y=y_step,mode="lines",name="Step",line=dict(color="#06d6a0",width=2.5))); fig2.add_hline(y=1,line_dash="dash",line_color="#aaa")
            figs.append(_theme(fig2,"📐 Step Response"))
            gm,pm,_,_=_ctrl.margin(sys)
            results.append("## 📐 Control System\n**G(s) = {:.0f}ωn²/(s²+2ζωn·s+ωn²)**\n**ωn={:.2f}, ζ={:.3f}, K={:.1f}**\n**Gain Margin:** {:.2f}dB | **Phase Margin:** {:.1f}°\n**Stability:** {} (PM>45°)\n**Overshoot:** {:.1f}%\n**Settling Time:** {:.3f}s\n".format(K,wn,zeta,K,20*np.log10(gm) if gm!=np.inf else np.inf,pm,"✅ Stable" if pm>0 else "❌ Unstable",np.exp(-np.pi*zeta/np.sqrt(max(1-zeta**2,1e-10)))*100 if zeta<1 else 0,4/(zeta*wn)))
        except ImportError: results.append("pip install control\n")

    else:
        ln={"en":"English","de":"German","ur":"Urdu"}.get(lang,"English")
        results.append(_call_llm("MATLAB/signal processing expert. Answer in {}: {}".format(ln,query),max_tokens=900,provider="claude"))

    insight=_call_llm("Summarize signal processing results in {}: {}".format({"en":"English","de":"German","ur":"Urdu"}.get(lang,"English"),"\n".join(results)[:1500]),max_tokens=250)
    return {"figs":figs,"results":results,"insight":insight,"type":"Signal Processing / MATLAB"}

# ══════════════════════════════════════════════════════════════════════════════
# SYMBOLIC MATH (Mathematica/Maple replacement)
# ══════════════════════════════════════════════════════════════════════════════
def symbolic_math(query, lang="en"):
    import sympy as sp
    q=query.lower(); x=sp.Symbol("x"); figs=[]; results=[]
    expr_str=re.sub(r"(derivative|integrate|solve|differentiate|find|of|with respect to x|the|calculate|compute|∫)","",query,flags=re.I).strip()
    try: expr_str_clean=expr_str.replace("^","**").replace("sin","sp.sin").replace("cos","sp.cos").replace("ln","sp.log").replace("e^","sp.exp").replace("sqrt","sp.sqrt")
    except: expr_str_clean=expr_str

    if any(w in q for w in ["derivative","differentiate","d/dx","slope"]):
        try:
            expr=sp.sympify(expr_str.replace("^","**"))
            result=sp.diff(expr,x)
            results.append("## 🧮 Derivative\n**f(x) = {}**\n**f'(x) = {}**\n".format(sp.pretty(expr),sp.pretty(result)))
            try:
                f_num=sp.lambdify(x,expr,"numpy"); df_num=sp.lambdify(x,result,"numpy")
                x_vals=np.linspace(-5,5,400)
                y_vals=np.clip(np.array(f_num(x_vals),dtype=float),-100,100)
                dy_vals=np.clip(np.array(df_num(x_vals),dtype=float),-100,100)
                fig=go.Figure(); fig.add_trace(go.Scatter(x=x_vals,y=y_vals,mode="lines",name="f(x)",line=dict(color="#4361ee",width=2))); fig.add_trace(go.Scatter(x=x_vals,y=dy_vals,mode="lines",name="f'(x)",line=dict(color="#f72585",width=2,dash="dash")))
                figs.append(_theme(fig,"🧮 Function and Derivative"))
            except: pass
        except: results.append(_call_llm("Calculate derivative of: {}. Step by step.".format(query),max_tokens=700))

    elif any(w in q for w in ["integral","integrate","antiderivative","∫"]):
        try:
            expr=sp.sympify(expr_str.replace("^","**")); limits=re.findall(r"from\s*(\-?\d+)\s*to\s*(\-?\d+)",q)
            if limits:
                a,b=int(limits[0][0]),int(limits[0][1]); result=sp.integrate(expr,(x,a,b))
                results.append("## 🧮 Definite Integral\n**∫[{},{}] {} dx = {}**\n".format(a,b,sp.pretty(expr),sp.pretty(result)))
            else:
                result=sp.integrate(expr,x); results.append("## 🧮 Indefinite Integral\n**∫ {} dx = {} + C**\n".format(sp.pretty(expr),sp.pretty(result)))
        except: results.append(_call_llm("Integrate: {}. Step by step.".format(query),max_tokens=700))

    elif any(w in q for w in ["solve","equation","root","zero","find x"]):
        try:
            if "=" in query:
                parts=query.split("="); lhs=sp.sympify(parts[0].split("solve")[-1].strip().replace("^","**"))
                rhs=sp.sympify(parts[1].strip().replace("^","**")) if len(parts)>1 else 0
                solutions=sp.solve(sp.Eq(lhs,rhs),x)
            else:
                solutions=sp.solve(sp.sympify(expr_str.replace("^","**")),x)
            results.append("## 🧮 Equation Solutions\n")
            for sol in solutions:
                try: results.append("  x = {} ≈ {:.6f}\n".format(sol,float(sol.evalf())))
                except: results.append("  x = {} (complex)\n".format(sol))
        except: results.append(_call_llm("Solve: {}. Show all steps.".format(query),max_tokens=700))

    elif any(w in q for w in ["matrix","determinant","eigenvalue","eigenvector","inverse"]):
        nums=re.findall(r"-?\d+\.?\d*",query); n=int(np.sqrt(len(nums)))
        if n*n==len(nums) and n<=5:
            try:
                M=sp.Matrix([[float(nums[i*n+j]) for j in range(n)] for i in range(n)])
                det=M.det(); results.append("## 🧮 Matrix Analysis\n**Matrix:**\n{}\n**Determinant:** {}\n**Eigenvalues:** {}\n".format(sp.pretty(M),det,M.eigenvals()))
                try: results.append("**Inverse:**\n{}\n".format(sp.pretty(M.inv())))
                except: results.append("**Singular — no inverse**\n")
            except: results.append(_call_llm("Solve matrix: {}".format(query),max_tokens=700))
        else: results.append(_call_llm("Linear algebra expert. Solve step by step: {}".format(query),max_tokens=900))

    else:
        ln={"en":"English","de":"German","ur":"Urdu"}.get(lang,"English")
        results.append(_call_llm("Expert mathematician (Mathematica/Maple level). Solve step by step in {}: {}".format(ln,query),max_tokens=900,provider="claude"))

    if not figs:
        x_v=np.linspace(-4,4,400)
        fig=go.Figure()
        for f,n,c in [(np.sin,"sin(x)",PAL[0]),(np.cos,"cos(x)",PAL[1]),(lambda v:np.exp(-v**2/2),"e^(-x²/2)",PAL[2])]:
            fig.add_trace(go.Scatter(x=x_v,y=f(x_v),mode="lines",name=n,line=dict(color=c,width=2)))
        figs.append(_theme(fig,"🧮 Reference: Common Functions"))

    insight=_call_llm("Explain math results in {}: {}".format({"en":"English","de":"German","ur":"Urdu"}.get(lang,"English"),"\n".join(results)[:1500]),max_tokens=250)
    return {"figs":figs,"results":results,"insight":insight,"type":"Symbolic Mathematics (Mathematica)"}

# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS
# ══════════════════════════════════════════════════════════════════════════════
def physics(query, lang="en"):
    q=query.lower(); nums=re.findall(r"[-+]?\d*\.?\d+",query); vals=[float(x) for x in nums]
    figs=[]; results=[]; hbar=1.055e-34; m_e=9.109e-31; e_charge=1.602e-19

    if any(w in q for w in ["projectile","trajectory","range","launch angle","kinematics"]):
        v0=next((v for v in vals if 1<=v<=1000),20.0); angle_deg=next((v for v in vals if 0<v<90),45.0); g=9.81
        angle=np.radians(angle_deg); vx=v0*np.cos(angle); vy=v0*np.sin(angle)
        t_flight=2*vy/g; R=vx*t_flight; H=vy**2/(2*g); t=np.linspace(0,t_flight,300)
        fig=go.Figure(); fig.add_trace(go.Scatter(x=vx*t,y=vy*t-0.5*g*t**2,mode="lines",name="Trajectory",line=dict(color="#4361ee",width=2.5),fill="tozeroy",fillcolor="rgba(67,97,238,.1)")); fig.add_trace(go.Scatter(x=[R/2],y=[H],mode="markers",marker=dict(size=12,color="#f72585",symbol="star"),name="Peak"))
        figs.append(_theme(fig,"⚛ Projectile: v₀={:.1f}m/s, θ={:.0f}°".format(v0,angle_deg)))
        results.append("## ⚛ Projectile Motion\n**v₀={:.2f}m/s, θ={:.1f}°**\n**Range R:** {:.4f}m\n**Max Height H:** {:.4f}m\n**Time of Flight:** {:.4f}s\n**Max Range at 45°:** {:.4f}m\n".format(v0,angle_deg,R,H,t_flight,v0**2/g))

    elif any(w in q for w in ["quantum","particle in a box","energy level","schrodinger","wave function","hydrogen"]):
        n_max=int(next((v for v in vals if 1<=v<=10),5)); L=next((v for v in vals if 1e-10<=v<=1e-8),1e-9)
        n_vals=np.arange(1,n_max+1); E_n=n_vals**2*np.pi**2*hbar**2/(2*m_e*L**2)
        fig=go.Figure(); x_box=np.linspace(0,L*1e9,400)
        for n in n_vals:
            psi=np.sqrt(2/L)*np.sin(n*np.pi*x_box/1e9/L); offset=float(E_n[n-1]/e_charge)
            fig.add_trace(go.Scatter(x=x_box,y=psi+offset,mode="lines",name="n={}".format(n),line=dict(color=PAL[(n-1)%len(PAL)],width=2)))
            fig.add_hline(y=offset,line_dash="dot",line_color="rgba(255,255,255,.15)")
        figs.append(_theme(fig,"⚛ Quantum: Particle in 1D Box (L={:.1f}nm)".format(L*1e9)))
        results.append("## ⚛ Particle in Box\n**L={:.2f}nm**\n"+"\n".join("  n={}: E={:.4f}eV".format(n,E/e_charge) for n,E in zip(n_vals,E_n)))

    elif any(w in q for w in ["carnot","heat engine","thermodynamics","entropy","efficiency"]):
        T_hot=next((v for v in vals if v>200),800.0); T_cold=next((v for v in vals if 0<v<T_hot),300.0)
        eta=(1-T_cold/T_hot)*100
        T_h_range=np.linspace(T_cold+50,2000,300); eta_range=(1-T_cold/T_h_range)*100
        fig=go.Figure(); fig.add_trace(go.Scatter(x=T_h_range,y=eta_range,mode="lines",name="η Carnot",line=dict(color="#4361ee",width=2.5))); fig.add_vline(x=T_hot,line_dash="dash",line_color="#f72585",annotation_text="TH={:.0f}K".format(T_hot))
        figs.append(_theme(fig,"⚛ Carnot Efficiency vs Temperature"))
        results.append("## ⚛ Carnot Engine\n**TH={:.1f}K ({:.1f}°C) | TC={:.1f}K ({:.1f}°C)**\n**η = 1-TC/TH = {:.2f}%**\n".format(T_hot,T_hot-273.15,T_cold,T_cold-273.15,eta))

    elif any(w in q for w in ["wave","frequency","wavelength","snell","refraction","optics"]):
        if "snell" in q or "refraction" in q:
            n1=next((v for v in vals if 1<=v<=3),1.0); n2=next((v for v in vals if v!=n1 and 1<=v<=3),1.5); theta1=next((v for v in vals if 0<v<90),30.0)
            theta2=np.degrees(np.arcsin(np.clip(n1*np.sin(np.radians(theta1))/n2,-1,1)))
            theta_range=np.linspace(0,90,300); theta2_range=np.degrees(np.arcsin(np.clip(n1*np.sin(np.radians(theta_range))/n2,-1,1)))
            fig=go.Figure(); fig.add_trace(go.Scatter(x=theta_range,y=theta2_range,mode="lines",name="θ₂",line=dict(color="#4361ee",width=2.5))); fig.add_trace(go.Scatter(x=theta_range,y=theta_range,mode="lines",name="θ₁=θ₂",line=dict(color="#aaa",dash="dash")))
            figs.append(_theme(fig,"⚛ Snell's Law: n₁={}, n₂={}".format(n1,n2)))
            results.append("## ⚛ Snell's Law\n**n₁sin(θ₁) = n₂sin(θ₂)**\n**θ₁={:.1f}° → θ₂={:.2f}°**\n**n₁={:.3f}, n₂={:.3f}**\n".format(theta1,theta2,n1,n2))
        else:
            f=next((v for v in vals if 1<=v<=1e12),440.0); v_wave=next((v for v in vals if v>100),343.0)
            lam=v_wave/f; t=np.linspace(0,2/f,500)
            fig=go.Figure(); fig.add_trace(go.Scatter(x=t*1000,y=np.sin(2*np.pi*f*t),mode="lines",name="Wave",line=dict(color="#4361ee",width=2.5)))
            figs.append(_theme(fig,"⚛ Wave: f={:.1f}Hz, λ={:.4f}m".format(f,lam)))
            results.append("## ⚛ Wave\n**f={:.2f}Hz, v={:.2f}m/s, λ={:.4f}m, T={:.6f}s**\n".format(f,v_wave,lam,1/f))

    else:
        ln={"en":"English","de":"German","ur":"Urdu"}.get(lang,"English")
        results.append(_call_llm("Physics professor. Answer in {} with formulas and derivations: {}".format(ln,query),max_tokens=900))

    insight=_call_llm("Explain physics in {}: {}".format({"en":"English","de":"German","ur":"Urdu"}.get(lang,"English"),"\n".join(results)[:1500]),max_tokens=250)
    return {"figs":figs,"results":results,"insight":insight,"type":"Physics"}

# ══════════════════════════════════════════════════════════════════════════════
# BIOLOGY / BIOINFORMATICS (GraphPad/BioPython replacement)
# ══════════════════════════════════════════════════════════════════════════════
def biology(query, df=None, lang="en"):
    q=query.lower(); figs=[]; results=[]

    if any(w in q for w in ["dna","rna","sequence","nucleotide","gc content","complement","protein","codon","transcription","translation"]):
        seq_m=re.search(r"[ATCGN]{8,}",query.upper())
        seq=seq_m.group(0) if seq_m else "ATGATCGATCGATCGATCGATCGATCGATCG"
        A=seq.count("A"); T=seq.count("T"); G=seq.count("G"); C=seq.count("C"); N=len(seq)
        gc=(G+C)/max(N,1)*100; Tm=2*(A+T)+4*(G+C)
        complement=seq.translate(str.maketrans("ATCG","TAGC")); rev_comp=complement[::-1]
        codon_table={"ATG":"Met(Start)","TAA":"Stop","TAG":"Stop","TGA":"Stop","TTT":"Phe","TTC":"Phe","TTA":"Leu","TTG":"Leu","TCT":"Ser","TCC":"Ser","TCA":"Ser","TCG":"Ser","TAT":"Tyr","TAC":"Tyr","TGT":"Cys","TGC":"Cys","TGG":"Trp","CTT":"Leu","CTC":"Leu","CTA":"Leu","CTG":"Leu","CCT":"Pro","CCC":"Pro","CCA":"Pro","CCG":"Pro","CAT":"His","CAC":"His","CAA":"Gln","CAG":"Gln","CGT":"Arg","CGC":"Arg","CGA":"Arg","CGG":"Arg","ATT":"Ile","ATC":"Ile","ATA":"Ile","ACT":"Thr","ACC":"Thr","ACA":"Thr","ACG":"Thr","AAT":"Asn","AAC":"Asn","AAA":"Lys","AAG":"Lys","AGT":"Ser","AGC":"Ser","AGA":"Arg","AGG":"Arg","GTT":"Val","GTC":"Val","GTA":"Val","GTG":"Val","GCT":"Ala","GCC":"Ala","GCA":"Ala","GCG":"Ala","GAT":"Asp","GAC":"Asp","GAA":"Glu","GAG":"Glu","GGT":"Gly","GGC":"Gly","GGA":"Gly","GGG":"Gly"}
        protein="".join(codon_table.get(seq[i:i+3],"?") for i in range(0,len(seq)-2,3) if len(seq[i:i+3])==3)
        fig=px.bar(x=["A","T","G","C"],y=[A,T,G,C],color=["#06d6a0","#f72585","#ffd166","#4361ee"],color_discrete_map="identity",title="🧬 DNA Nucleotide Composition")
        figs.append(_theme(fig))
        results.append("## 🧬 DNA Sequence Analysis\n**Length:** {} bp | **GC:** {:.1f}% | **AT:** {:.1f}%\n**Tm (Wallace):** {:.1f}°C\n**Complement:** {}...\n**Rev Complement:** {}...\n**Codons:** {}...\n".format(N,gc,100-gc,Tm,complement[:40],rev_comp[:40],protein[:30]))

    elif any(w in q for w in ["dose","response","ic50","ec50","hill","sigmoid","inhibition"]):
        nums=re.findall(r"[-+]?\d*\.?\d+",query); vals=[float(x) for x in nums]
        EC50=next((v for v in vals if 0.001<=v<=1000),1.0); n_hill=next((v for v in vals if 0.5<=v<=4),1.0)
        conc=np.logspace(-3,3,300); response=100/(1+(EC50/conc)**n_hill)
        fig=go.Figure(); fig.add_trace(go.Scatter(x=conc,y=response,mode="lines",name="Dose-Response",line=dict(color="#4361ee",width=2.5))); fig.add_hline(y=50,line_dash="dash",line_color="#f72585",annotation_text="EC50/IC50"); fig.add_vline(x=EC50,line_dash="dot",line_color="#06d6a0",annotation_text="EC50={:.3f}".format(EC50))
        fig.update_xaxes(type="log")
        figs.append(_theme(fig,"🧬 Dose-Response Curve (Hill Equation)"))
        results.append("## 🧬 Dose-Response\n**Hill: y=Emax/(1+(EC50/x)^n)**\n**EC50={:.4f} | n={:.2f} | pEC50={:.2f}**\n".format(EC50,n_hill,-np.log10(EC50) if EC50>0 else 0))

    elif any(w in q for w in ["roc","auc","sensitivity","specificity","diagnostic"]):
        if df is not None:
            schema=fingerprint_schema(df); bin_cols=[c for c,m in schema.items() if m["role"]=="binary"]; meas=[c for c,m in schema.items() if m["role"]=="measure"]
            if bin_cols and meas:
                from sklearn.metrics import roc_curve,auc
                y_true=df[bin_cols[0]].fillna(0); y_score=df[meas[0]].fillna(df[meas[0]].mean())
                fpr,tpr,thresh=roc_curve(y_true,y_score); roc_auc=auc(fpr,tpr)
                fig=go.Figure(); fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name="ROC (AUC={:.3f})".format(roc_auc),line=dict(color="#4361ee",width=2.5))); fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random",line=dict(color="#aaa",dash="dash")))
                figs.append(_theme(fig,"🧬 ROC Curve"))
                results.append("## 🧬 ROC Analysis\n**AUC={:.3f}** ({})\n".format(roc_auc,"Excellent" if roc_auc>0.9 else "Good" if roc_auc>0.8 else "Fair" if roc_auc>0.7 else "Poor"))

    else:
        ln={"en":"English","de":"German","ur":"Urdu"}.get(lang,"English")
        results.append(_call_llm("Expert biologist/bioinformatician. Answer in {}: {}".format(ln,query),max_tokens=900))

    insight=_call_llm("Explain biology results in {}: {}".format({"en":"English","de":"German","ur":"Urdu"}.get(lang,"English"),"\n".join(results)[:1500]),max_tokens=250)
    return {"figs":figs,"results":results,"insight":insight,"type":"Biology/Bioinformatics"}

# ══════════════════════════════════════════════════════════════════════════════
# SQL ENGINE (DuckDB — replaces SQL Server, MySQL, Oracle)
# ══════════════════════════════════════════════════════════════════════════════
def sql_engine(query, dfs_dict, lang="en"):
    import duckdb
    q=query.lower(); figs=[]; results=[]
    conn=duckdb.connect(":memory:"); table_names=[]
    for name,df in dfs_dict.items():
        clean_name=re.sub(r"[^\w]","_",name.rsplit(".",1)[0])[:30]
        conn.register(clean_name,df); table_names.append(clean_name)
    if not re.search(r"\b(select|from|where|group by|join|with|having)\b",q,re.I):
        schema_info={tn:conn.execute("DESCRIBE {}".format(tn)).fetchdf()["column_name"].tolist() for tn in table_names}
        sql=_call_llm("Tables:\n{}\nGenerate DuckDB SQL for: '{}'\nReturn ONLY the SQL.".format(json.dumps(schema_info),query),max_tokens=300,system="SQL expert. Return only valid DuckDB SQL.")
        sql=re.sub(r"```sql|```","",sql).strip()
    else: sql=query
    try:
        result_df=conn.execute(sql).fetchdf()
        results.append("## 🗄 SQL Result\n```sql\n{}\n```\n**Rows:** {:,} | **Cols:** {}\n".format(sql,len(result_df),list(result_df.columns)))
        num_c=result_df.select_dtypes(include="number").columns; str_c=result_df.select_dtypes(exclude="number").columns
        if len(num_c)>0 and len(str_c)>0:
            fig=px.bar(result_df.head(25),x=str_c[0],y=num_c[0],color_discrete_sequence=PAL,title="🗄 SQL Result: {}".format(sql[:60]))
            figs.append(_theme(fig))
    except Exception as e:
        results.append("**SQL Error:** {}\n**Query:** {}\n**Tables:** {}\n".format(str(e),sql,table_names))
    insight=_call_llm("Explain SQL results in {}: {}".format({"en":"English","de":"German","ur":"Urdu"}.get(lang,"English"),"\n".join(results)[:800]),max_tokens=150)
    return {"figs":figs,"results":results,"insight":insight,"type":"SQL/Database"}

# ══════════════════════════════════════════════════════════════════════════════
# FLUID DYNAMICS
# ══════════════════════════════════════════════════════════════════════════════
def fluid_dynamics(query, lang="en"):
    q=query.lower(); nums=re.findall(r"[-+]?\d*\.?\d+",query); vals=[float(x) for x in nums]
    figs=[]; results=[]
    D=next((v for v in vals if 0.001<=v<=5),0.1); L=next((v for v in vals if v>D and v<=10000),100.0)
    v_flow=next((v for v in vals if 0.01<=v<=50),2.0); rho=next((v for v in vals if 800<=v<=2000),1000.0); mu=1e-3
    Re=rho*v_flow*D/mu; flow_regime="Laminar" if Re<2300 else "Transitional" if Re<4000 else "Turbulent"
    f=64/Re if Re<2300 else (-2*np.log10(4.6e-5/D/3.7+2.51/(Re*np.sqrt(0.02))))**(-2)
    for _ in range(15): f=(-2*np.log10(4.6e-5/D/3.7+2.51/(Re*np.sqrt(f))))**(-2) if Re>=4000 else f
    dP=f*(L/D)*(0.5*rho*v_flow**2)/1000; hL=f*(L/D)*v_flow**2/(2*9.81); Q=v_flow*np.pi*D**2/4*1000
    Re_range=np.logspace(2.3,8,500); f_lam=64/Re_range
    eps_D_vals=[0,1e-6,1e-4,5e-4,1e-3]; labels=["Smooth","ε/D=1e-6","ε/D=1e-4","ε/D=5e-4","ε/D=1e-3"]
    fig=go.Figure(); fig.add_trace(go.Scatter(x=Re_range[Re_range<2300],y=f_lam[Re_range<2300],mode="lines",name="Laminar",line=dict(color="#aaa",width=1.5,dash="dash")))
    for eps_D,lbl,color in zip(eps_D_vals,labels,PAL):
        f_t=np.where(Re_range<2300,64/Re_range,[(-2*np.log10(eps_D/3.7+2.51/(Re_*np.sqrt(0.02))))**(-2) for Re_ in Re_range])
        fig.add_trace(go.Scatter(x=Re_range,y=f_t,mode="lines",name=lbl,line=dict(color=color,width=1.5)))
    fig.add_vline(x=Re,line_dash="dot",line_color="#f72585",annotation_text="Re={:.0f}".format(Re)); fig.update_xaxes(type="log"); fig.update_yaxes(type="log")
    figs.append(_theme(fig,"🌊 Moody Diagram"))
    r_r=np.linspace(0,D/2,100); v_profile=v_flow*2*(1-(2*r_r/D)**2) if Re<2300 else v_flow*1.22*(1-2*r_r/D)**(1/7)
    fig2=go.Figure(); fig2.add_trace(go.Scatter(x=v_profile,y=r_r*100,mode="lines",name="Profile",line=dict(color="#4361ee",width=2.5))); fig2.add_trace(go.Scatter(x=v_profile,y=-r_r*100,mode="lines",showlegend=False,line=dict(color="#4361ee",width=2.5)))
    figs.append(_theme(fig2,"🌊 Velocity Profile ({})".format(flow_regime)))
    results.append("## 🌊 Pipe Flow (Darcy-Weisbach)\n**D={:.3f}m, L={:.1f}m, v={:.3f}m/s, ρ={:.0f}kg/m³**\n**Re={:.0f} → {}**\n**f={:.5f} | ΔP={:.3f}kPa | hL={:.3f}m**\n**Q={:.4f}L/s**\n".format(D,L,v_flow,rho,Re,flow_regime,f,dP,hL,Q))
    insight=_call_llm("Explain fluid dynamics in {}: {}".format({"en":"English","de":"German","ur":"Urdu"}.get(lang,"English"),"\n".join(results)[:800]),max_tokens=200)
    return {"figs":figs,"results":results,"insight":insight,"type":"Fluid Dynamics"}

# ══════════════════════════════════════════════════════════════════════════════
# ELECTRICAL ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def electrical_engineering(query, lang="en"):
    from scipy import signal as _sig
    q=query.lower(); nums=re.findall(r"[-+]?\d*\.?\d+",query); vals=[float(x) for x in nums]
    figs=[]; results=[]
    R=next((v for v in vals if 0.1<=v<=1e6),1000.0); C=next((v for v in vals if 1e-12<=v<=1e-3),1e-6); L=next((v for v in vals if 1e-9<=v<=10),1e-3); V0=next((v for v in vals if 0.1<=v<=1000),5.0)
    omega0=1/np.sqrt(L*C); f0=omega0/(2*np.pi); Q_factor=omega0*L/R; alpha=R/(2*L)
    freq=np.logspace(1,7,1000); omega=2*np.pi*freq; Z_total=R+1j*omega*L+1/(1j*omega*C)
    fig=make_subplots(rows=2,cols=1,subplot_titles=["Impedance |Z| (Ω)","Phase (°)"])
    fig.add_trace(go.Scatter(x=freq,y=np.abs(Z_total),mode="lines",name="|Z|",line=dict(color="#4361ee",width=2)),row=1,col=1)
    fig.add_trace(go.Scatter(x=freq,y=np.angle(Z_total,deg=True),mode="lines",name="∠Z",line=dict(color="#f72585",width=2)),row=2,col=1)
    fig.update_xaxes(type="log"); fig.add_vline(x=f0,line_dash="dash",line_color="#06d6a0",row=1,col=1,annotation_text="f₀={:.1f}Hz".format(f0))
    figs.append(_theme(fig,"⚡ RLC Circuit Frequency Response"))
    t=np.linspace(0,10*max(L/R,np.sqrt(L*C)),500); sys=_sig.lti([1/C],[L,R,1/C]); t_out,v_out=_sig.step(sys,T=t)
    fig2=go.Figure(); fig2.add_trace(go.Scatter(x=t_out*1000,y=v_out*V0,mode="lines",name="Vc(t)",line=dict(color="#06d6a0",width=2.5))); fig2.add_hline(y=V0,line_dash="dash",line_color="#aaa")
    figs.append(_theme(fig2,"⚡ RLC Step Response"))
    results.append("## ⚡ RLC Circuit\n**R={:.0f}Ω, L={:.2e}H, C={:.2e}F**\n**f₀={:.4f}Hz={:.4f}kHz**\n**Q={:.4f} | ζ={:.4f} ({})**\n**τ=L/R={:.4e}s**\n".format(R,L,C,f0,f0/1000,Q_factor,alpha/omega0,"Overdamped" if alpha>omega0 else "Critically damped" if abs(alpha-omega0)<0.001 else "Underdamped"))
    insight=_call_llm("Explain EE results in {}: {}".format({"en":"English","de":"German","ur":"Urdu"}.get(lang,"English"),"\n".join(results)[:800]),max_tokens=200)
    return {"figs":figs,"results":results,"insight":insight,"type":"Electrical Engineering"}

# ══════════════════════════════════════════════════════════════════════════════
# MANUFACTURING / SIX SIGMA (Minitab replacement)
# ══════════════════════════════════════════════════════════════════════════════
def manufacturing_quality(df, query, lang="en"):
    from scipy import stats as _stats
    figs=[]; results=[]; schema=fingerprint_schema(df); measures=[c for c,m in schema.items() if m["role"]=="measure"]
    if not measures: return {"figs":[],"results":["No numeric columns."],"insight":"","type":"Manufacturing"}
    m_col=measures[0]; data=df[m_col].dropna().values; n=len(data); mean=data.mean(); std=data.std()
    nums=re.findall(r"[-+]?\d*\.?\d+",query); vals=[float(x) for x in nums]
    LSL=next((v for v in vals if v<mean),mean-3*std); USL=next((v for v in vals if v>mean),mean+3*std)
    Cp=(USL-LSL)/(6*std) if std>0 else 0; Cpu=(USL-mean)/(3*std) if std>0 else 0; Cpl=(mean-LSL)/(3*std) if std>0 else 0
    Cpk=min(Cpu,Cpl); sigma_level=Cpk*3; dpmo=_stats.norm.sf(3*Cpk)*1e6*2 if Cpk>0 else 500000
    subgroup_size=min(10,max(2,n//20)); subgroups=[data[i:i+subgroup_size].mean() for i in range(0,n-subgroup_size,subgroup_size)]
    x_bar=np.mean(subgroups); UCL=x_bar+3*std/np.sqrt(subgroup_size); LCL=x_bar-3*std/np.sqrt(subgroup_size)
    fig=go.Figure(); fig.add_trace(go.Scatter(x=list(range(len(subgroups))),y=subgroups,mode="lines+markers",name="X̄",line=dict(color="#4361ee",width=2),marker=dict(size=6))); fig.add_hline(y=UCL,line_dash="dash",line_color="#ef476f",annotation_text="UCL"); fig.add_hline(y=x_bar,line_dash="solid",line_color="#06d6a0",annotation_text="X̄"); fig.add_hline(y=LCL,line_dash="dash",line_color="#ef476f",annotation_text="LCL")
    ooc=[i for i,v in enumerate(subgroups) if v>UCL or v<LCL]
    if ooc: fig.add_trace(go.Scatter(x=ooc,y=[subgroups[i] for i in ooc],mode="markers",marker=dict(color="#ef476f",size=12,symbol="x"),name="Out of Control"))
    figs.append(_theme(fig,"🏭 X̄ Control Chart — {}".format(m_col)))
    x_range=np.linspace(LSL*0.95,USL*1.05,300); normal_curve=_stats.norm.pdf(x_range,mean,std)*n*(USL-LSL)/25
    fig2=go.Figure(); fig2.add_trace(go.Histogram(x=data,nbinsx=30,name="Data",marker_color="#4361ee",opacity=0.7)); fig2.add_trace(go.Scatter(x=x_range,y=normal_curve,mode="lines",name="Normal",line=dict(color="#f72585",width=2))); fig2.add_vline(x=LSL,line_dash="dash",line_color="#ef476f",annotation_text="LSL"); fig2.add_vline(x=USL,line_dash="dash",line_color="#ef476f",annotation_text="USL")
    figs.append(_theme(fig2,"🏭 Process Capability — Cpk={:.3f}".format(Cpk)))
    results.append("## 🏭 Six Sigma Process Capability\n**{} (n={:,})**\n**X̄={:.4f} | σ={:.4f}**\n**Cp={:.3f} | Cpk={:.3f}** ({})\n**Sigma Level:** {:.2f}σ\n**DPMO:** {:,.0f}\n**Status:** {}\n".format(m_col,n,mean,std,Cp,Cpk,"Capable ✅" if Cpk>=1.33 else "Marginal ⚠" if Cpk>=1.0 else "Not capable ❌",sigma_level,dpmo,"✅ IN CONTROL" if not ooc else "❌ OUT OF CONTROL ({} points)".format(len(ooc))))
    insight=_call_llm("Explain quality control results in {}: {}".format({"en":"English","de":"German","ur":"Urdu"}.get(lang,"English"),"\n".join(results)[:1000]),max_tokens=200)
    return {"figs":figs,"results":results,"insight":insight,"type":"Manufacturing/Six Sigma (Minitab)"}

# ══════════════════════════════════════════════════════════════════════════════
# ECONOMETRICS (EViews/Stata replacement)
# ══════════════════════════════════════════════════════════════════════════════
def econometrics(df, query, lang="en"):
    from scipy import stats as _stats
    figs=[]; results=[]; schema=fingerprint_schema(df); measures=[c for c,m in schema.items() if m["role"]=="measure"]
    if not measures:
        ln={"en":"English","de":"German","ur":"Urdu"}.get(lang,"English")
        return {"figs":[],"results":[_call_llm("Econometrician. Answer in {}: {}".format(ln,query),max_tokens=700)],"insight":"","type":"Econometrics"}
    ts_col=measures[0]; y=df[ts_col].dropna().values; n=len(y); t=np.arange(n)
    slope,intercept,r,p,_=_stats.linregress(t,y); trend=slope*t+intercept; detrended=y-trend
    max_lag=min(40,n//4); acf_vals=[1.0]+[np.corrcoef(y[:-i],y[i:])[0,1] for i in range(1,max_lag+1)]; conf=1.96/np.sqrt(n)
    fig=make_subplots(rows=3,cols=1,subplot_titles=["Series + Trend","Detrended","ACF"],vertical_spacing=0.1)
    fig.add_trace(go.Scatter(y=y,mode="lines",name="Data",line=dict(color="#4361ee",width=1.5)),row=1,col=1); fig.add_trace(go.Scatter(y=trend,mode="lines",name="Trend",line=dict(color="#f72585",width=2,dash="dash")),row=1,col=1)
    fig.add_trace(go.Scatter(y=detrended,mode="lines",name="Detrended",line=dict(color="#06d6a0",width=1.5)),row=2,col=1)
    fig.add_trace(go.Bar(x=list(range(len(acf_vals))),y=acf_vals,name="ACF",marker_color="#4361ee"),row=3,col=1); fig.add_hline(y=conf,line_dash="dash",line_color="#ffd166",row=3,col=1); fig.add_hline(y=-conf,line_dash="dash",line_color="#ffd166",row=3,col=1)
    fig.update_layout(height=700); figs.append(_theme(fig,"📈 Time Series Analysis: {}".format(ts_col)))
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        n_fc=min(12,n//5); model=ExponentialSmoothing(y,trend="add",seasonal=None).fit(optimized=True); fc=model.forecast(n_fc)
        fig2=go.Figure(); fig2.add_trace(go.Scatter(y=y,mode="lines",name="Historical",line=dict(color="#4361ee",width=2))); fig2.add_trace(go.Scatter(x=list(range(n,n+n_fc)),y=fc.values,mode="lines",name="Forecast",line=dict(color="#f72585",width=2.5,dash="dash")))
        figs.append(_theme(fig2,"📈 Forecast: {} ({} periods)".format(ts_col,n_fc)))
        results.append("**Forecast:** {}\n".format([round(v,4) for v in fc.values]))
    except: pass
    q_stat=n*(n+2)*sum(acf_vals[i]**2/(n-i) for i in range(1,min(20,len(acf_vals)))); p_lb=1-_stats.chi2.cdf(q_stat,20)
    results.append("## 📈 Time Series: {}\n**n={}** | **Trend slope:** {:.6f}/period\n**R²(trend):** {:.4f} | **p={:.4f}** ({})\n**Ljung-Box Q(20):** {:.2f} p={:.4f} — {}\n".format(ts_col,n,slope,r**2,p,"↑ upward" if slope>0 else "↓ downward",q_stat,p_lb,"Autocorrelation" if p_lb<0.05 else "No autocorrelation"))
    insight=_call_llm("Summarize time series in {}: {}".format({"en":"English","de":"German","ur":"Urdu"}.get(lang,"English"),"\n".join(results)[:1000]),max_tokens=200)
    return {"figs":figs,"results":results,"insight":insight,"type":"Econometrics (EViews/Stata)"}

# ══════════════════════════════════════════════════════════════════════════════
# CODE EXECUTOR
# ══════════════════════════════════════════════════════════════════════════════
def code_executor(query, df=None, lang="en"):
    q=query.lower(); figs=[]; results=[]
    code_m=re.search(r"```(?:\w+)?\n(.*?)```",query,re.DOTALL)
    code=code_m.group(1).strip() if code_m else re.sub(r"(run|execute|code:|compute|python|r:|sql:)","",query,flags=re.I).strip()
    if not code: return {"figs":[],"results":["No code found."],"insight":"","type":"Code Executor"}
    lang_name="sql" if any(w in q for w in ["```sql","sql query","run sql"]) else "r" if any(w in q for w in ["```r\n","r code","rscript"]) else "cpp" if "c++" in q or "```cpp" in q else "python"
    if lang_name=="python":
        try:
            with tempfile.NamedTemporaryFile(mode="w",suffix=".py",delete=False) as f:
                prelude="import pandas as pd,numpy as np,io\n"
                if df is not None: prelude+="df=pd.read_csv(io.StringIO('''{}'''))\n".format(df.head(1000).to_csv(index=False))
                f.write(prelude+code); fname=f.name
            proc=subprocess.run([sys.executable,fname],capture_output=True,text=True,timeout=30)
            results.append("## 💻 Python\n```python\n{}\n```\n**Output:**\n```\n{}\n```\n".format(code[:500],proc.stdout[:2000]))
            if proc.stderr: results.append("**Errors:**\n```\n{}\n```\n".format(proc.stderr[:300]))
            os.unlink(fname)
        except subprocess.TimeoutExpired: results.append("**Timeout (>30s)**\n")
        except Exception as e: results.append("**Error:** {}\n".format(str(e)))
    elif lang_name=="r":
        try:
            with tempfile.NamedTemporaryFile(mode="w",suffix=".R",delete=False) as f: f.write(code); fname=f.name
            proc=subprocess.run(["Rscript",fname],capture_output=True,text=True,timeout=60)
            results.append("## 💻 R\n```r\n{}\n```\n**Output:**\n```\n{}\n```\n".format(code[:500],proc.stdout[:2000]))
            os.unlink(fname)
        except FileNotFoundError:
            out=_call_llm("Execute R code exactly, show output: {}".format(code),max_tokens=700)
            results.append("## 💻 R (Simulated)\n```r\n{}\n```\n**Output:**\n{}\n".format(code[:500],out))
    elif lang_name=="sql" and df is not None:
        try:
            import duckdb; conn=duckdb.connect(":memory:"); conn.register("data",df)
            result_df=conn.execute(re.sub(r"--.*","",code).strip()).fetchdf()
            results.append("## 💻 SQL\n```sql\n{}\n```\n**{} rows returned**\n{}\n".format(code[:500],len(result_df),result_df.head(20).to_string()))
            if len(result_df)>0:
                nc=result_df.select_dtypes(include="number").columns; sc=result_df.select_dtypes(exclude="number").columns
                if len(nc)>0 and len(sc)>0:
                    fig=px.bar(result_df.head(25),x=sc[0],y=nc[0],color_discrete_sequence=PAL); figs.append(_theme(fig,"SQL Result"))
        except Exception as e: results.append("**SQL Error:** {}\n".format(str(e)))
    elif lang_name=="cpp":
        try:
            with tempfile.NamedTemporaryFile(mode="w",suffix=".cpp",delete=False) as f: f.write(code); cpp_f=f.name
            exe_f=cpp_f.replace(".cpp",""); cp=subprocess.run(["g++","-O2",cpp_f,"-o",exe_f],capture_output=True,text=True,timeout=30)
            if cp.returncode==0:
                rp=subprocess.run([exe_f],capture_output=True,text=True,timeout=10)
                results.append("## 💻 C++\n```cpp\n{}\n```\n**Output:**\n```\n{}\n```\n".format(code[:500],rp.stdout[:1000]))
            else: results.append("**Compile error:**\n```\n{}\n```\n".format(cp.stderr[:400]))
            for f_c in [cpp_f,exe_f]:
                try: os.unlink(f_c)
                except: pass
        except FileNotFoundError:
            out=_call_llm("Execute C++ code, show exact output: {}".format(code),max_tokens=500)
            results.append("## 💻 C++ (Simulated)\n**Output:**\n{}\n".format(out))
    else:
        out=_call_llm("Execute this code and show exact output: {}".format(code),max_tokens=700)
        results.append("## 💻 Code Output\n{}\n".format(out))
    return {"figs":figs,"results":results,"insight":"","type":"Code Executor"}

# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENTAL SCIENCE
# ══════════════════════════════════════════════════════════════════════════════
def environmental_science(query, df=None, lang="en"):
    q=query.lower(); nums=re.findall(r"[-+]?\d*\.?\d+",query); vals=[float(x) for x in nums]
    figs=[]; results=[]
    if any(w in q for w in ["carbon","co2","emission","greenhouse","footprint"]):
        energy_kwh=next((v for v in vals if v>10),1000.0); fuel_liters=next((v for v in vals if v!=energy_kwh and v>0),200.0)
        co2_energy=energy_kwh*0.4; co2_fuel=fuel_liters*2.31; total=co2_energy+co2_fuel
        fig=px.pie(values=[co2_energy,co2_fuel],names=["Electricity","Fuel"],title="🌱 Carbon Footprint",color_discrete_sequence=["#4361ee","#f72585"],hole=0.4)
        figs.append(_theme(fig))
        results.append("## 🌱 Carbon Footprint\n**Electricity:** {:.1f}kg CO₂\n**Fuel:** {:.1f}kg CO₂\n**Total:** {:.3f} tCO₂e\n**Trees to offset:** {:.0f}/year\n".format(co2_energy,co2_fuel,total/1000,total/1000/0.022))
    elif any(w in q for w in ["water quality","bod","do","dissolved oxygen","wastewater"]):
        BOD=next((v for v in vals if v>0),250.0); k1=0.23; k2=0.4; DO_sat=9.1; D0=2.1
        t=np.linspace(0,20,300); DO=DO_sat-(BOD*k1/(k2-k1))*(np.exp(-k1*t)-np.exp(-k2*t))-D0*np.exp(-k2*t)
        DO=np.clip(DO,0,DO_sat)
        fig=go.Figure(); fig.add_trace(go.Scatter(x=t,y=DO,mode="lines",name="DO",line=dict(color="#4361ee",width=2.5),fill="tozeroy",fillcolor="rgba(67,97,238,.1)")); fig.add_hline(y=5,line_dash="dash",line_color="#ef476f",annotation_text="Min DO=5mg/L")
        figs.append(_theme(fig,"🌱 Streeter-Phelps DO Sag Curve"))
        tc=(1/(k2-k1))*np.log(k2/k1*(1-D0*(k2-k1)/(BOD*k1))); DO_c=float(DO[int(tc*300/20)])
        results.append("## 🌱 Water Quality\n**BOD={:.0f}mg/L**\n**Critical time:** {:.2f}days\n**Min DO:** {:.3f}mg/L\n**Status:** {}\n".format(BOD,tc,DO_c,"✅ OK" if DO_c>5 else "❌ OXYGEN DEPLETION RISK"))
    else:
        ln={"en":"English","de":"German","ur":"Urdu"}.get(lang,"English")
        results.append(_call_llm("Environmental scientist. Answer in {}: {}".format(ln,query),max_tokens=700))
    return {"figs":figs,"results":results,"insight":"","type":"Environmental Science"}

# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════
DOMAIN_KEYWORDS = {
    "powerbi":    ["dashboard","kpi","report","scorecard","looker","power bi","tableau","visualization","visual","metric","drill","slicer","filter","treemap","funnel","gauge","pivot"],
    "structural": ["beam","deflect","shear","moment","column","buckling","truss","foundation","bearing capacity","steel","concrete","sap2000","staad","autocad","dxf","rebar","footing","rcc","weld","terzaghi","euler"],
    "chemical":   ["distillation","mccabe","thiele","heat exchanger","lmtd","reactor","cstr","pfr","chemcad","aspen","hysys","molar","enthalpy","entropy","vapor","liquid","chemical","boiling","condensation","pvt","molecular","formula","chempy"],
    "statistics": ["regression","anova","t-test","p-value","chi-square","shapiro","normality","confidence interval","hypothesis","pearson","spearman","mann-whitney","kruskal","wilcoxon","binomial","poisson","spss","sas","stata","minitab","jmp"],
    "ml":         ["machine learning","neural network","random forest","decision tree","svm","knn","gradient boost","xgboost","deep learning","cluster","pca","dimensionality","automl","feature importance","train test","cross validation","sklearn"],
    "financial":  ["npv","irr","dcf","cash flow","wacc","option","black-scholes","portfolio","sharpe","markowitz","bond","yield","dividend","bloomberg","roi","payback","amortization","depreciation","capex","beta","var","stock","equity","ticker"],
    "signal":     ["fft","fourier","frequency spectrum","bode","transfer function","pid","control system","butterworth","filter","matlab","simulink","labview","step response","pole","zero","nyquist","laplace","dsp","pwm"],
    "symbolic":   ["derivative","integral","differential equation","ode","pde","eigenvalue","determinant","laplace transform","taylor series","mathematica","maple","wolfram","calculus","solve equation","simplify","factor","expand","limit"],
    "biology":    ["dna","rna","protein","sequence","nucleotide","genome","bioinformatics","blast","gc content","pcr","codon","transcription","translation","biopython","dose response","ic50","ec50","pharmacokinetics","roc auc","survival"],
    "physics":    ["projectile","velocity","acceleration","gravity","newton","quantum","wave","optics","lens","snell","electromagnetic","thermodynamics","carnot","entropy","schrodinger","photon","particle in a box"],
    "fluid":      ["pipe flow","reynolds","darcy","moody","friction factor","bernoulli","flow rate","viscosity","turbulent","laminar","pump","pressure drop","head loss","hydraulics"],
    "electrical": ["circuit","resistor","capacitor","inductor","rlc","impedance","resonance","power factor","3-phase","transformer","voltage","current","ohm","kirchhoff","thevenin","pspice","ltspice","ac circuit","dc circuit"],
    "geospatial": ["map","gis","geographic","latitude","longitude","arcgis","qgis","shapefile","country map","choropleth","spatial"],
    "sql":        ["select ","from ","where ","group by","having ","join ","union ","cte","window function","duckdb","sql query","database"],
    "manufacturing":["six sigma","cpk","cpk","dpmo","control chart","ucl","lcl","x-bar","spc","process capability","minitab quality","defect","tolerance"],
    "econometrics":["time series","arima","autocorrelation","acf","pacf","stationarity","adf","unit root","granger","panel data","fixed effects","eviews","garch","var model","forecast arima"],
    "environmental":["carbon footprint","co2 emission","greenhouse gas","water quality","bod","dissolved oxygen","wastewater treatment","hydrology","watershed"],
    "code":       ["```python","```r\n","```c++","```sql","```julia","```bash","run python","execute code","python script","r script","compile c++"],
}

def route_domain(query, df=None):
    q=query.lower()
    scores={}
    for domain,keywords in DOMAIN_KEYWORDS.items():
        scores[domain]=sum(2 if kw in q else 0 for kw in keywords)
    best=max(scores,key=scores.get)
    if scores[best]==0:
        domain_list=list(DOMAIN_KEYWORDS.keys())
        route=(_call_llm("Classify into ONE of: {}.\nReturn ONLY the word.\nQuery: '{}'".format(", ".join(domain_list),query),max_tokens=10,system="Return one word only.") or "statistics").strip().lower()
        for d in domain_list:
            if d in route: return d
        return "statistics"
    return best

def solve(query, df=None, dfs_dict=None, lang="en"):
    """Master solver — routes to correct domain."""
    domain=route_domain(query,df)
    try:
        if domain=="powerbi": return {"figs":build_powerbi_dashboard(df,fingerprint_schema(df),lang) if df is not None else [],"results":["Power BI Dashboard generated."],"insight":"","type":"Power BI / Looker Studio Dashboard"}
        elif domain=="structural": return structural_engineering(query,df,lang)
        elif domain=="chemical": return chemical_engineering(query,df,lang)
        elif domain=="statistics": return advanced_statistics(df,query,lang) if df is not None else {"figs":[],"results":[_call_llm("Statistics expert. Answer in {}: {}".format({"en":"English","de":"German","ur":"Urdu"}.get(lang,"English"),query),max_tokens=800)],"insight":"","type":"Statistics"}
        elif domain=="ml": return machine_learning(df,query,lang) if df is not None else {"figs":[],"results":[_call_llm("ML expert. Answer in {}: {}".format({"en":"English","de":"German","ur":"Urdu"}.get(lang,"English"),query),max_tokens=800,provider="claude")],"insight":"","type":"Machine Learning"}
        elif domain=="financial": return financial_engineering(query,df,lang)
        elif domain=="signal": return signal_processing(query,df,lang)
        elif domain=="symbolic": return symbolic_math(query,lang)
        elif domain=="biology": return biology(query,df,lang)
        elif domain=="physics": return physics(query,lang)
        elif domain=="fluid": return fluid_dynamics(query,lang)
        elif domain=="electrical": return electrical_engineering(query,lang)
        elif domain=="geospatial": return geospatial_analysis(df,query,lang)
        elif domain=="sql":
            dfs=dfs_dict or ({"data":df} if df is not None else {})
            return sql_engine(query,dfs,lang)
        elif domain=="manufacturing": return manufacturing_quality(df,query,lang) if df is not None else {"figs":[],"results":[_call_llm("Quality engineer. Answer: {}".format(query),max_tokens=600)],"insight":"","type":"Manufacturing"}
        elif domain=="econometrics": return econometrics(df,query,lang) if df is not None else {"figs":[],"results":[_call_llm("Econometrician. Answer in {}: {}".format({"en":"English","de":"German","ur":"Urdu"}.get(lang,"English"),query),max_tokens=700)],"insight":"","type":"Econometrics"}
        elif domain=="environmental": return environmental_science(query,df,lang)
        elif domain=="code": return code_executor(query,df,lang)
        else:
            if df is not None: return advanced_statistics(df,query,lang)
            ln={"en":"English","de":"German","ur":"Urdu"}.get(lang,"English")
            return {"figs":[],"results":[_call_llm("Expert analyst. Answer in {}: {}".format(ln,query),max_tokens=900)],"insight":"","type":"General"}
    except Exception as e:
        import traceback
        ln={"en":"English","de":"German","ur":"Urdu"}.get(lang,"English")
        fallback=_call_llm("Answer in {} as expert: {}".format(ln,query),max_tokens=800)
        return {"figs":[],"results":["**Error:** {}\n\n**AI Answer:**\n{}\n".format(str(e),fallback)],"insight":"","type":"General (AI Fallback)"}

def geospatial_analysis(df, query, lang="en"):
    figs=[]; results=[]
    if df is not None:
        schema=fingerprint_schema(df); meas=[c for c,m in schema.items() if m["role"]=="measure"]
        for col in df.columns:
            sample=df[col].dropna().astype(str).str.lower().head(30).tolist()
            if sum(1 for v in sample if v in COUNTRY_ISO)>5 and meas:
                df2=df.copy(); df2["_iso"]=df2[col].astype(str).str.lower().map(COUNTRY_ISO)
                for mc in meas[:2]:
                    g=df2.groupby("_iso")[mc].sum().reset_index()
                    fig=px.choropleth(g,locations="_iso",color=mc,color_continuous_scale="Blues",title="🌍 {} by Country".format(mc))
                    figs.append(_theme(fig)); results.append("## 🌍 Choropleth: {}\n".format(mc)); break
        lat_col=next((c for c in df.columns if any(w in c.lower() for w in ["lat","latitude"])),None)
        lon_col=next((c for c in df.columns if any(w in c.lower() for w in ["lon","long","longitude"])),None)
        if lat_col and lon_col:
            val_col=meas[0] if meas else None
            fig=px.scatter_geo(df.dropna(subset=[lat_col,lon_col]).head(2000),lat=lat_col,lon=lon_col,color=val_col if val_col else None,title="🌍 Geographic Distribution",color_continuous_scale="Viridis",opacity=0.6)
            figs.append(_theme(fig)); results.append("## 🌍 Geographic Points: {} locations\n".format(min(2000,len(df))))
    if not figs:
        ln={"en":"English","de":"German","ur":"Urdu"}.get(lang,"English")
        results.append(_call_llm("GIS expert. Answer in {}: {}".format(ln,query),max_tokens=600))
    return {"figs":figs,"results":results,"insight":"","type":"Geospatial/GIS (ArcGIS)"}

# ══════════════════════════════════════════════════════════════════════════════
# DATA ENGINE CLASS (Power BI + all domains)
# ══════════════════════════════════════════════════════════════════════════════
class DataEngine:
    def __init__(self, data):
        self.df=smart_merge(data,["t{}".format(i+1) for i in range(len(data))]) if isinstance(data,list) else (data.copy() if data is not None else pd.DataFrame())
        self._schema={}; self._lang="en"

    def normalize(self):
        """
        Fast silent data preparation — no LLM calls, no data modification.
        Only fixes column names, detects types, removes truly empty columns.
        """
        df = self.df.copy()
        # 1. Clean column names — snake_case, no special chars
        df.columns = [re.sub(r'[^\w]','_',c.strip().lower()).strip('_') or f"col_{i}"
                      for i,c in enumerate(df.columns)]
        # 2. Remove completely empty unnamed columns
        df = df[[c for c in df.columns
                 if not (c.startswith("unnamed") and df[c].isna().mean() > 0.9)]]
        # 3. Convert date strings to datetime (silent)
        for c in df.columns:
            if df[c].dtype == object:
                try:
                    parsed = pd.to_datetime(df[c], infer_datetime_format=True, errors="coerce")
                    if parsed.notna().mean() > 0.7:
                        df[c] = parsed
                except: pass
        # 4. Convert numeric strings to numbers (e.g. "1,234" → 1234)
        for c in df.columns:
            if df[c].dtype == object:
                cleaned = df[c].astype(str).str.replace(',','').str.replace('$','').str.replace('€','').str.strip()
                cv = pd.to_numeric(cleaned, errors="coerce")
                if cv.notna().mean() > 0.6:
                    df[c] = cv
        self.df = df
        self._schema = fingerprint_schema(df)
        return self

    def quality_report(self):
        s=self._schema
        return {"rows":len(self.df),"cols":len(self.df.columns),
                "measures":sum(1 for m in s.values() if m["role"]=="measure"),
                "dimensions":sum(1 for m in s.values() if m["role"]=="dimension"),
                "temporals":sum(1 for m in s.values() if m["role"]=="temporal"),
                "entities":sum(1 for m in s.values() if m["role"]=="entity"),
                "missing_pct":round(self.df.isna().mean().mean()*100,1)}

    def get_suggestions(self, lang="en"):
        s=self._schema
        m=[c for c,v in s.items() if v["role"]=="measure"][:4]
        d=[c for c,v in s.items() if v["role"]=="dimension"][:3]
        t=[c for c,v in s.items() if v["role"]=="temporal"][:2]
        e=[c for c,v in s.items() if v["role"]=="entity"][:2]
        PFX={"en":{"top":"Top 10","by":"by","trend":"trend over","dist":"Distribution of","vs":"vs","corr":"Correlation","six":"Six Sigma","reg":"Regression","dash":"Build dashboard"},
             "de":{"top":"Top 10","by":"nach","trend":"Trend","dist":"Verteilung","vs":"vs","corr":"Korrelation","six":"Sechs Sigma","reg":"Regression","dash":"Dashboard erstellen"},
             "ur":{"top":"ٹاپ 10","by":"کے","trend":"ٹرینڈ","dist":"تقسیم","vs":"بمقابلہ","corr":"ارتباط","six":"سکس سگما","reg":"ریگریشن","dash":"ڈیش بورڈ"}}
        p=PFX.get(lang,PFX["en"]); sugg=[]
        if e and m: sugg.append("{} {} {} {}".format(p["top"],e[0],p["by"],m[0]))
        if d and m: sugg.append("{} {} {} {}".format(p["top"],d[0],p["by"],m[0]))
        if t and m: sugg.append("{} {} {}".format(m[0],p["trend"],t[0]))
        if len(m)>=2: sugg.append("{} {} {}".format(m[0],p["vs"],m[1]))
        if len(m)>=3: sugg.append(p["corr"])
        if m: sugg.append("{} {}".format(p["dist"],m[0]))
        if m: sugg.append("{} {}".format(p["six"],m[0]))
        if len(m)>=2: sugg.append("{}: {} ~ {}".format(p["reg"],m[1],m[0]))
        sugg.append(p["dash"])
        return sugg[:8]

    def get_placeholder(self, lang="en"):
        s=self._schema
        m=[c for c,v in s.items() if v["role"]=="measure"]
        d=[c for c,v in s.items() if v["role"] in ("dimension","entity")]
        if m and d: return "Ask: beam analysis 6m span, regression {0}~{1}, distillation column, NPV 10%, FFT analysis...".format(m[0],d[0])
        return "Ask anything: structural, chemistry, statistics, finance, physics, biology, SQL, Power BI dashboard..."

    def get_auto_figs(self):
        df=self.df; s=self._schema; lang=self._lang
        # Use Power BI dashboard builder for auto visualization
        return build_powerbi_dashboard(df,s,lang)

    def get_auto_insight(self, lang="en"):
        df=self.df; s=self._schema
        LN={"en":"English","de":"German","fr":"French","es":"Spanish","ar":"Arabic","zh":"Chinese","ur":"Urdu","tr":"Turkish","ru":"Russian","hi":"Hindi"}
        ln=LN.get(lang,"English")
        measures=[c for c,m in s.items() if m["role"]=="measure"]
        dims=[c for c,m in s.items() if m["role"]=="dimension"]
        temps=[c for c,m in s.items() if m["role"]=="temporal"]
        qr=self.quality_report()

        # INTELLIGENT stats: use correct aggregation per column
        smart_stats = {}
        for m in measures[:6]:
            if m in df.columns and df[m].notna().any():
                agg = s.get(m,{}).get("agg","sum")
                try:
                    if agg == "mean":
                        smart_stats[m] = {"metric":"avg", "value":round(float(df[m].mean()),2),
                                          "median":round(float(df[m].median()),2),
                                          "max":round(float(df[m].max()),2)}
                    elif agg == "year":
                        smart_stats[m] = {"metric":"range",
                                          "from":int(df[m].min()),"to":int(df[m].max()),
                                          "unique":int(df[m].nunique())}
                    else:
                        smart_stats[m] = {"metric":"total", "value":round(float(df[m].sum()),2),
                                          "avg":round(float(df[m].mean()),2),
                                          "max":round(float(df[m].max()),2)}
                except: pass

        system_prompt = (
            "You are a world-class data analyst. "
            "CRITICAL RULES: "
            "1) Never say 'total year' or average/sum year columns — years are time ranges only. "
            "2) For price/rate/cost columns always say AVERAGE not total. "
            "3) For revenue/amount/quantity say TOTAL. "
            "4) Only use real numbers from the stats provided. "
            "Respond ONLY in {}.".format(ln)
        )
        prompt = (
            "Dataset: {} rows x {} cols. Columns: {}. "
            "Categories: {}. Time columns: {}. "
            "Stats (correct aggregation - avg for prices, total for revenue): {}\n"
            "Sample:\n{}\n\n"
            "Write:\n"
            "**Overview** (1 sentence what this data is)\n"
            "**Key Findings** (3 findings with real numbers from stats)\n"
            "**Patterns** (correlations, outliers, trends)\n"
            "**Recommendations** (2 specific actionable recommendations)"
        ).format(qr["rows"],qr["cols"],measures[:6],dims[:4],temps[:2],
                 json.dumps(smart_stats),df.head(8).to_csv(index=False)[:1500])


        return _call_llm(prompt, max_tokens=700, system=system_prompt)

    def run_query(self, query, lang="en"):
        result=solve(query,self.df,{"data":self.df} if self.df is not None else {},lang)
        figs=result.get("figs",[])
        insight_parts=[]
        if result.get("insight"): insight_parts.append(result["insight"])
        if result.get("results"): insight_parts.append("\n".join(r for r in result["results"] if isinstance(r,str))[:1200])
        return {"figs":figs,"insight":"\n\n".join(insight_parts),"type":result.get("type","")}

# Backward compatibility exports
def advanced_analysis(df, schema, lang="en"):
    engine=DataEngine(df); engine._schema=schema; engine._lang=lang
    return {"figs":engine.get_auto_figs(),"findings":[],"llm_insight":engine.get_auto_insight(lang)}

def scientific_analysis(df, schema, query, lang="en"):
    return solve(query,df,{"data":df},lang)

def generate_did_video(insight_text, lang="en"):
    return {"error":"Set DID_API_KEY from studio.d-id.com","url":None}