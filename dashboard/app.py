"""
Zynetrax v5.0 — dashboard/app.py
4 sections: Search · Analyze · Jobs · Chat
Cinematic presentation: 3D avatar + Pexels videos + Ken Burns + Charts
"""
import io, os, json, base64, re, subprocess, sys, tempfile
import pandas as pd

import dash
from dash import html, dcc, Input, Output, State, ctx, ALL, no_update, dash_table
from dash.exceptions import PreventUpdate

from intelligence.search_engine import SearchEngine
from intelligence.data_engine import (
    DataEngine, _call_llm, generate_did_video,
    advanced_analysis, scientific_analysis, fingerprint_schema, smart_merge
)
from openai import OpenAI

_ds  = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY",""), base_url="https://api.deepseek.com")
_oai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY",""))

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY","")

def _call_claude_app(prompt, max_tokens=1024, system=""):
    """Claude API — precise chart specs, analysis, structured output."""
    try:
        import urllib.request
        payload = json.dumps({"model":"claude-sonnet-4-5","max_tokens":max_tokens,
            "system":system or "You are a world-class data analyst. Follow instructions exactly.",
            "messages":[{"role":"user","content":prompt}]}).encode()
        req = urllib.request.Request("https://api.anthropic.com/v1/messages",data=payload,
            headers={"x-api-key":ANTHROPIC_KEY,"anthropic-version":"2023-06-01","content-type":"application/json"})
        with urllib.request.urlopen(req,timeout=30) as resp:
            return json.loads(resp.read())["content"][0]["text"].strip()
    except Exception: return None


PEXELS_KEY = os.environ.get("PEXELS_API_KEY", "")
RPM_AVATAR = os.environ.get("RPM_AVATAR_URL", "")  # Ready Player Me .glb URL

# ── colors ─────────────────────────────────────────────────────────────────────
BG="#0a0a14";CARD="#12121f";BDR="#1e1e35";PRI="#4361ee"
ACC="#f72585";SUC="#06d6a0";WARN="#ffd166";DNG="#ef476f"
TXT="#e8e8f0";MUT="#8888aa";HDR="#07070f";ACC2="#4cc9f0"
PAL=["#4361ee","#f72585","#4cc9f0","#7209b7","#06d6a0","#ffd166","#ef476f","#118ab2","#26a269","#e76f51"]

def btn(bg=PRI,fg="white",extra=None):
    s={"backgroundColor":bg,"color":fg,"border":"none","borderRadius":"8px","padding":"10px 20px",
       "fontSize":"14px","fontWeight":"600","cursor":"pointer","fontFamily":"Segoe UI,Arial",
       "transition":"all .15s","height":"44px"}
    if extra: s.update(extra)
    return s

def inp(extra=None):
    s={"padding":"10px 14px","borderRadius":"8px","background":"#1a1a2e","border":"1.5px solid "+BDR,
       "color":TXT,"fontSize":"14px","fontFamily":"Segoe UI,Arial","outline":"none",
       "width":"100%","boxSizing":"border-box","height":"44px"}
    if extra: s.update(extra)
    return s

def card(extra=None):
    s={"background":CARD,"borderRadius":"14px","padding":"22px 26px","marginBottom":"20px",
       "border":"1px solid "+BDR,"boxShadow":"0 4px 24px rgba(0,0,0,.4)"}
    if extra: s.update(extra)
    return s

def h2(t): return html.Div(t,style={"fontSize":"18px","fontWeight":"700","color":TXT,"marginBottom":"16px"})

def ibox(text):
    if not text: return html.Div()
    return html.Div([html.Span("🔮 "),
        dcc.Markdown(str(text),style={"fontSize":"13px","lineHeight":"1.7","color":TXT,"display":"inline"})],
        style={"background":"rgba(67,97,238,.08)","borderLeft":"4px solid "+PRI,
               "padding":"14px 18px","borderRadius":"8px","marginBottom":"12px","border":"1px solid rgba(67,97,238,.2)"})

def chart_card(fig, num=None):
    try: h=int(fig.layout.height or 400)
    except: h=400
    return html.Div([
        html.Div("Chart {}".format(num),style={"fontSize":"10px","color":MUT,"marginBottom":"4px","fontWeight":"600"}) if num else html.Div(),
        dcc.Graph(figure=fig,style={"height":"{}px".format(max(360,h)),"width":"100%"},
                  config={"displayModeBar":True,"displaylogo":False,"scrollZoom":True,
                          "modeBarButtonsToRemove":["select2d","lasso2d"]}),
    ],style={"background":CARD,"borderRadius":"12px","padding":"10px 14px","border":"1px solid "+BDR,"marginBottom":"8px","id":"chart-card-{}".format(num or 0)})

def grid(figs):
    if not figs: return html.Div()
    cs=[chart_card(f,i+1) for i,f in enumerate(figs)]
    if len(cs)==1: return cs[0]
    if len(cs)==2: return html.Div([html.Div(c,style={"width":"49%"}) for c in cs],style={"display":"flex","gap":"2%"})
    return html.Div([cs[0],html.Div([html.Div(cs[i],style={"width":"49%"}) for i in range(1,len(cs))],
                    style={"display":"flex","gap":"2%","flexWrap":"wrap","marginTop":"8px"})])

# ── LANGUAGES ──────────────────────────────────────────────────────────────────
LANGS={
    "en":"English","de":"Deutsch","fr":"Français","es":"Español","ar":"العربية",
    "zh":"中文","hi":"हिन्दी","pt":"Português","ru":"Русский","ja":"日本語",
    "ko":"한국어","it":"Italiano","nl":"Nederlands","tr":"Türkçe","ur":"اردو",
    "fa":"فارسی","pl":"Polski","sv":"Svenska","da":"Dansk","fi":"Suomi",
    "nb":"Norsk","cs":"Čeština","ro":"Română","hu":"Magyar","el":"Ελληνικά",
    "he":"עברית","uk":"Українська","vi":"Tiếng Việt","th":"ภาษาไทย","id":"Bahasa Indonesia",
    "ms":"Bahasa Melayu","bn":"বাংলা","ta":"தமிழ்","sw":"Kiswahili","af":"Afrikaans",
    "tl":"Filipino","sk":"Slovenčina","bg":"Български","hr":"Hrvatski","lt":"Lietuvių",
}
LN={"en":"English","de":"German","fr":"French","es":"Spanish","ar":"Arabic","zh":"Chinese",
    "hi":"Hindi","pt":"Portuguese","ru":"Russian","ja":"Japanese","ko":"Korean","it":"Italian",
    "nl":"Dutch","tr":"Turkish","ur":"Urdu","fa":"Persian","pl":"Polish","sv":"Swedish",
    "da":"Danish","fi":"Finnish","nb":"Norwegian","cs":"Czech","ro":"Romanian","hu":"Hungarian",
    "el":"Greek","he":"Hebrew","uk":"Ukrainian","vi":"Vietnamese","th":"Thai","id":"Indonesian",
    "ms":"Malay","bn":"Bengali","ta":"Tamil","sw":"Swahili","af":"Afrikaans","tl":"Filipino",
    "sk":"Slovak","bg":"Bulgarian","hr":"Croatian","lt":"Lithuanian"}

TABS=["search","analyze","chat"]
TLABEL={"search":"🔍 Search","analyze":"📊 Analyze","chat":"💬 Chat"}

# ══════════════════════════════════════════════════════════════════════════════
# CINEMATIC PRESENTATION ENGINE — Pure HTML/CSS/JS, zero cost
# ══════════════════════════════════════════════════════════════════════════════
PRESENTATION_HTML = ""  # Removed — Coming Soon feature


# ── CSS + JS ────────────────────────────────────────────────────────────────────
CSS="""
body{margin:0;background:#0a0a14;color:#e8e8f0;font-family:'Segoe UI',Arial,sans-serif}
*{box-sizing:border-box}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-thumb{background:#2a2a4a;border-radius:3px}
.tab-btn{background:transparent;color:#8888aa;border:1px solid #1e1e35;border-radius:8px;
  padding:8px 16px;font-size:13px;font-weight:600;cursor:pointer;white-space:nowrap;
  transition:all .15s;font-family:'Segoe UI',Arial;margin:2px}
.tab-btn:hover{background:rgba(67,97,238,.2);color:#4cc9f0;border-color:#4361ee}
.tab-active{background:#4361ee!important;color:white!important;border-color:#4361ee!important}
.tool-pill-active{background:rgba(67,97,238,.25)!important;box-shadow:0 0 12px rgba(67,97,238,.4)}
[id*="tool-pill"]:hover{opacity:.85;transform:translateY(-1px)}
@keyframes spin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}
@keyframes pulse-glow{0%,100%{opacity:1}50%{opacity:.5}}
.processing-msg{display:flex;align-items:center;gap:10px;padding:10px 14px;
  background:rgba(67,97,238,.08);border:1px solid rgba(67,97,238,.2);
  border-radius:10px;margin-top:8px}
.processing-spinner{width:18px;height:18px;border:2px solid #4361ee;
  border-top-color:transparent;border-radius:50%;animation:spin .8s linear infinite;flex-shrink:0}
.processing-text{color:#4cc9f0;font-size:13px;font-weight:600}
.done-msg{display:flex;align-items:center;gap:8px;color:#06d6a0;font-size:13px;font-weight:600}
.job-card{background:#12121f;border-radius:12px;padding:16px 20px;margin-bottom:10px;
  border:1px solid #1e1e35;transition:border-color .15s}
.job-card:hover{border-color:#4361ee}
.code-editor{background:#0d1117;border:1.5px solid #1e1e35;border-radius:8px;padding:14px;
  font-family:'Courier New',monospace;font-size:13px;color:#e6edf3;width:100%;
  min-height:200px;resize:vertical;outline:none;box-sizing:border-box}
.code-editor:focus{border-color:#4361ee}
._dash-loading-callback,._dash-loading,.dash-debug-menu,.dash-error-card{display:none!important}
.Select-control,.Select-menu-outer{background:#1a1a2e!important;border-color:#1e1e35!important}
.Select-value-label,.VirtualizedSelectOption{color:#e8e8f0!important;background:#1a1a2e!important}
.VirtualizedSelectFocusedOption{background:#2a2a4e!important}
#zt-modal{display:none;position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,.85);
  align-items:center;justify-content:center}
#zt-modal.open{display:flex}
#zt-modal-box{background:#12121f;border-radius:20px;padding:40px;width:420px;
  max-width:95vw;box-shadow:0 24px 80px rgba(0,0,0,.7);border:1px solid #1e1e35;position:relative}
"""

JS="""
(function(){
  var _c=194000+Math.floor(Math.random()*8000),_d=1;
  setInterval(function(){
    var el=document.getElementById('live-count');if(!el)return;
    var s=1+Math.floor(Math.random()*3);_c+=_d*s;
    if(_c>216000)_d=-1;if(_c<170000)_d=1;
    if(Math.random()<.02)_d*=-1;
    el.textContent=_c.toLocaleString();
  },2500);
  var SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  var VL={en:'en-US',de:'de-DE',fr:'fr-FR',es:'es-ES',ar:'ar-SA',zh:'zh-CN',
    hi:'hi-IN',pt:'pt-BR',ru:'ru-RU',ja:'ja-JP',ko:'ko-KR',it:'it-IT',tr:'tr-TR',ur:'ur-PK'};
  window.voiceFor=function(id,lang){
    if(!SR){alert('Voice input needs Chrome or Edge');return;}
    var r=new SR();r.lang=VL[lang]||'en-US';r.interimResults=true;
    r.onresult=function(e){
      var t='';for(var i=0;i<e.results.length;i++)t+=e.results[i][0].transcript;
      var el=document.getElementById(id);if(!el)return;
      var d=Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el),'value');
      if(d&&d.set)d.set.call(el,t);else el.value=t;
      el.dispatchEvent(new Event('input',{bubbles:true}));
      el.dispatchEvent(new Event('change',{bubbles:true}));
    };r.start();
  };
  window.openModal=function(){var m=document.getElementById('zt-modal');if(m)m.classList.add('open');};
  window.closeModal=function(){var m=document.getElementById('zt-modal');if(m)m.classList.remove('open');};
  window.switchView=function(v){
    var lv=document.getElementById('zt-lv'),sv=document.getElementById('zt-sv');
    if(!lv||!sv)return;
    if(v==='signup'){lv.style.display='none';sv.style.display='block';}
    else{sv.style.display='none';lv.style.display='block';}
  };
})();
"""

INDEX="""<!DOCTYPE html>
<html>
<head>{%metas%}<title>Zynetrax — AI Data Intelligence Platform</title>
{%favicon%}{%css%}<style>"""+CSS+"""</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
<script>"""+JS+"""</script>
</body>
</html>"""

# ── helpers ─────────────────────────────────────────────────────────────────────
def _read_excel(buf):
    try:
        raw=pd.read_excel(buf,header=None,nrows=25,engine="openpyxl")
        best=max(range(min(len(raw),20)),
                 key=lambda i:sum(1 for v in raw.iloc[i] if isinstance(v,str) and len(str(v))>1),default=0)
        buf.seek(0)
        df=pd.read_excel(buf,header=best,engine="openpyxl")
        df=df[[c for c in df.columns if not(str(c).lower().startswith("unnamed") and df[c].isna().mean()>.7)]]
        for c in df.columns:
            cv=pd.to_numeric(df[c],errors="coerce")
            if cv.notna().mean()>.5: df[c]=cv
        return df
    except Exception:
        buf.seek(0)
        try: return pd.read_excel(buf,engine="xlrd")
        except Exception: return pd.read_csv(io.StringIO(buf.read().decode("utf-8","replace")))

def _read_csv(text):
    import csv as _c
    if isinstance(text,bytes): text=text.decode("utf-8","replace")
    try: sep=_c.Sniffer().sniff("\n".join(text.splitlines()[:15]),",\t|;").delimiter
    except: sep=","
    df=pd.read_csv(io.StringIO(text),sep=sep,on_bad_lines="skip",low_memory=False)
    for c in df.columns:
        cv=pd.to_numeric(df[c].astype(str).str.replace(",",""),errors="coerce")
        if cv.notna().mean()>.6: df[c]=cv
    return df

def _chat_llm(msgs, lang="en"):
    ln=LN.get(lang,"English")
    system={"role":"system","content":"You are Zynetrax AI — advanced data intelligence assistant. Respond ONLY in {}.".format(ln)}
    full=[system]+msgs[-14:]
    try:
        r=_ds.chat.completions.create(model="deepseek-chat",messages=full,max_tokens=700,temperature=0.7)
        return r.choices[0].message.content.strip()
    except Exception:
        pass
    try:
        r=_oai.chat.completions.create(model="gpt-4o-mini",messages=full,max_tokens=700,temperature=0.7)
        return r.choices[0].message.content.strip()
    except Exception as e:
        err=str(e)
        if "401" in err: return "⚠ API key error."
        return "⚠ Connection error."

def run_dashboard():
    app=dash.Dash(__name__,suppress_callback_exceptions=True)
    app.index_string=INDEX
    app.server.config["MAX_CONTENT_LENGTH"]=500*1024*1024

    # Prevent browser from caching old layout/callbacks
    @app.server.after_request
    def add_no_cache(response):
        response.headers["Cache-Control"]="no-cache, no-store, must-revalidate"
        response.headers["Pragma"]="no-cache"
        response.headers["Expires"]="0"
        return response
    se=SearchEngine()

    app.layout=html.Div(style={"background":BG,"minHeight":"100vh"},children=[
        dcc.Store(id="active-tab",data="search"),
        dcc.Store(id="lang",data="en"),
        dcc.Store(id="df-store"),
        dcc.Store(id="figs-store",data=[]),
        dcc.Store(id="insight-store",data=""),
        dcc.Store(id="chat-hist",data=[]),
        dcc.Store(id="page-n",data=1),
        dcc.Store(id="user-store",storage_type="session"),
        dcc.Store(id="cv-store"),
        # Tool selector store — top level
        dcc.Store(id="active-tool",data="powerbi"),
        dcc.Store(id="staged-files",data=[]),

        # ── MODAL ────────────────────────────────────────────────────────────
        html.Div(id="zt-modal",children=[
            html.Div(id="zt-modal-box",children=[
                html.Button("×",id="modal-close",n_clicks=0,
                    style={"position":"absolute","top":"12px","right":"16px","fontSize":"22px",
                           "cursor":"pointer","color":MUT,"background":"none","border":"none"}),
                html.Div(id="zt-lv",children=[
                    html.H2("Welcome to Zynetrax",style={"color":TXT,"fontSize":"22px","margin":"0 0 20px"}),
                    dcc.Input(id="l-email",type="email",placeholder="Email",style={**inp(),"marginBottom":"10px","display":"block"}),
                    dcc.Input(id="l-pw",type="password",placeholder="Password",style={**inp(),"marginBottom":"10px","display":"block"}),
                    html.Div(id="l-err",style={"color":DNG,"fontSize":"12px","marginBottom":"8px"}),
                    html.Button("Sign In",id="l-sub",n_clicks=0,style={**btn(),"width":"100%","marginBottom":"12px"}),
                    html.Div(["No account? ",html.Span("Sign up",id="to-signup",style={"color":PRI,"cursor":"pointer","textDecoration":"underline"})],
                             style={"textAlign":"center","fontSize":"13px","color":MUT}),
                ]),
                html.Div(id="zt-sv",style={"display":"none"},children=[
                    html.H2("Create Account",style={"color":TXT,"fontSize":"22px","margin":"0 0 20px"}),
                    dcc.Input(id="s-name",type="text",placeholder="Full name",style={**inp(),"marginBottom":"10px","display":"block"}),
                    dcc.Input(id="s-email",type="email",placeholder="Email",style={**inp(),"marginBottom":"10px","display":"block"}),
                    dcc.Input(id="s-pw",type="password",placeholder="Password (8+)",style={**inp(),"marginBottom":"10px","display":"block"}),
                    html.Div(id="s-err",style={"color":DNG,"fontSize":"12px","marginBottom":"8px"}),
                    html.Button("Create Account",id="s-sub",n_clicks=0,style={**btn(SUC,"#0a0a14"),"width":"100%","marginBottom":"12px"}),
                    html.Div(["Have account? ",html.Span("Sign in",id="to-login",style={"color":PRI,"cursor":"pointer","textDecoration":"underline"})],
                             style={"textAlign":"center","fontSize":"13px","color":MUT}),
                ]),
            ]),
        ]),

        # ── HEADER ───────────────────────────────────────────────────────────
        html.Div(style={"background":HDR,"borderBottom":"1px solid "+BDR,
                        "position":"sticky","top":"0","zIndex":"100","padding":"10px 16px"},children=[
            html.Div(style={"maxWidth":"1400px","margin":"0 auto","display":"flex",
                            "alignItems":"center","gap":"12px","flexWrap":"wrap"},children=[
                html.Div([html.Span("◈ ",style={"color":PRI,"fontSize":"22px"}),
                          html.Span("Zynetrax",style={"fontWeight":"800","fontSize":"20px","color":"white"})],
                         style={"flexShrink":"0","display":"flex","alignItems":"center"}),
                html.Div(style={"display":"flex","gap":"4px","flex":"1"},children=[
                    html.Button(TLABEL[t],id="tab-"+t,n_clicks=0,
                        className="tab-btn"+(" tab-active" if t=="search" else ""))
                    for t in TABS
                ]),
                html.Div(style={"display":"flex","alignItems":"center","gap":"10px","flexShrink":"0"},children=[
                    html.Div([html.Span("● ",style={"color":SUC,"fontSize":"8px"}),
                              html.Span(id="live-count",children="194,832",
                                  style={"color":SUC,"fontSize":"12px","fontWeight":"700","fontFamily":"monospace"}),
                              html.Span(" online",style={"color":MUT,"fontSize":"11px"})],
                             style={"background":"rgba(6,214,160,.08)","borderRadius":"20px","padding":"4px 10px","border":"1px solid rgba(6,214,160,.2)"}),
                    dcc.Dropdown(id="lang-dd",options=[{"label":v,"value":k} for k,v in LANGS.items()],
                        value="en",clearable=False,style={"width":"130px","fontSize":"12px"}),
                    html.Button("Sign In",id="login-btn",n_clicks=0,
                        style=btn(extra={"height":"34px","padding":"6px 16px","fontSize":"13px"})),
                ]),
            ]),
        ]),

        html.Div(id="main",style={"maxWidth":"1400px","margin":"0 auto","padding":"24px 16px"}),
    ])

    # ── ROUTING ───────────────────────────────────────────────────────────────
    @app.callback(Output("active-tab","data"),[Input("tab-"+t,"n_clicks") for t in TABS],prevent_initial_call=True)
    def set_tab(*args):
        t=ctx.triggered_id
        if t and t.startswith("tab-"): return t[4:]
        return no_update

    @app.callback([Output("tab-"+t,"className") for t in TABS],Input("active-tab","data"))
    def style_tabs(active): return ["tab-btn tab-active" if t==active else "tab-btn" for t in TABS]

    @app.callback(Output("lang","data"),Input("lang-dd","value"))
    def set_lang(v): return v or "en"

    app.clientside_callback("function(n){if(n>0)window.openModal&&window.openModal();return '';}",Output("login-btn","title"),Input("login-btn","n_clicks"),prevent_initial_call=True)
    app.clientside_callback("function(n){if(n>0)window.closeModal&&window.closeModal();return '';}",Output("modal-close","title"),Input("modal-close","n_clicks"),prevent_initial_call=True)
    app.clientside_callback("function(n){window.switchView&&window.switchView('signup');return '';}",Output("to-signup","title"),Input("to-signup","n_clicks"),prevent_initial_call=True)
    app.clientside_callback("function(n){window.switchView&&window.switchView('login');return '';}",Output("to-login","title"),Input("to-login","n_clicks"),prevent_initial_call=True)

    # ── INSTANT UPLOAD FEEDBACK (clientside = fires before server responds) ──
    # Processing indicator handled via dcc.Loading


    @app.callback(Output("user-store","data"),Output("l-err","children"),
                  Input("l-sub","n_clicks"),State("l-email","value"),State("l-pw","value"),prevent_initial_call=True)
    def login(n,email,pw):
        import random as _r
        if not email or not pw: return no_update,"Fill all fields."
        name=" ".join(w.capitalize() for w in email.split("@")[0].replace("."," ").split())
        return {"email":email,"name":name,"initials":"".join([w[0].upper() for w in name.split()[:2]]),"color":_r.choice(PAL)},""

    @app.callback(Output("user-store","data",allow_duplicate=True),Output("s-err","children"),
                  Input("s-sub","n_clicks"),State("s-name","value"),State("s-email","value"),State("s-pw","value"),prevent_initial_call=True)
    def signup(n,name,email,pw):
        import random as _r
        if not name or not email or not pw: return no_update,"Fill all fields."
        if len(pw or "")<8: return no_update,"Password must be 8+ chars."
        return {"email":email,"name":name,"initials":"".join([w[0].upper() for w in (name or "Z X").split()[:2]]),"color":_r.choice(PAL)},""

    # ── MAIN RENDER ───────────────────────────────────────────────────────────
    @app.callback(Output("main","children"),
                  Input("active-tab","data"),Input("lang","data"),
                  State("df-store","data"),State("chat-hist","data"),
                  State("user-store","data"),State("active-tool","data"))
    def render(tab,lang,df_data,chat_hist,user,active_tool):
        active_tool = active_tool or "powerbi"
        tab=tab or "search"; lang=lang or "en"

        # ── SEARCH ────────────────────────────────────────────────────────────
        if tab=="search":
            TAGS=["GDP growth","Climate change","Crime statistics","Population data",
                  "Stock prices","Health indicators","Energy data","Education index",
                  "Unemployment rates","Trade data","Inflation","COVID-19 data",
                  "Military spending","Poverty index","Food security","AI research"]
            return html.Div([
                html.Div(style=card(),children=[
                    html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"center","marginBottom":"14px"},children=[
                        h2("🔍 Search Any Dataset — 60+ Global Sources"),
                        html.Span("Unlimited results",style={"fontSize":"11px","color":SUC,"background":"rgba(6,214,160,.1)","padding":"3px 10px","borderRadius":"20px","border":"1px solid rgba(6,214,160,.2)"}),
                    ]),
                    html.Div(style={"display":"flex","gap":"8px","marginBottom":"12px"},children=[
                        dcc.Input(id="sq",type="text",debounce=False,n_submit=0,
                            placeholder="Search any topic: GDP, crime, climate, health, energy...",
                            style=inp({"flex":"1"})),
                        dcc.Dropdown(id="stype",
                            options=[{"label":x,"value":x} for x in ["ALL","CSV","Excel","JSON","PDF","Image","Video"]],
                            value="ALL",clearable=False,style={"width":"110px","fontSize":"13px","flexShrink":"0"}),
                        html.Button("🔍 Search",id="sbtn",n_clicks=0,style=btn(extra={"flexShrink":"0"})),
                        html.Button("Next →",id="snext",n_clicks=0,style=btn(BG,PRI,{"border":"1.5px solid "+PRI,"flexShrink":"0"})),
                    ]),
                    html.Div(style={"display":"flex","flexWrap":"wrap","gap":"5px"},children=[
                        html.Button(t,id={"type":"qtag","index":i},n_clicks=0,
                            style=btn("rgba(67,97,238,.1)",ACC2,{"border":"1px solid rgba(76,201,240,.2)","fontSize":"11px","padding":"4px 10px","height":"26px"}))
                        for i,t in enumerate(TAGS)
                    ]),
                ]),
                dcc.Loading(type="circle",color=PRI,children=html.Div(id="sout")),
            ])

        # ── ANALYZE ───────────────────────────────────────────────────────────
        if tab=="analyze":
            has_data = bool(df_data)

            # ── 5 ACTIVE TOOLS ────────────────────────────────────────────
            ACTIVE = [
                {"id":"powerbi","icon":"📊","label":"Power BI",
                 "color":"#f2c811","bg":"rgba(242,200,17,.1)",
                 "desc":"Dashboards · KPIs · Charts · Maps · Natural language queries",
                 "chips":["Build full dashboard","Top 10 by value","Trend over time",
                          "Correlation matrix","Distribution analysis","KPI scorecard"]},
                {"id":"python","icon":"🐍","label":"Python",
                 "color":"#4cc9f0","bg":"rgba(76,201,240,.1)",
                 "desc":"Data analysis · ML · Automation · Visualization · No coding needed",
                 "chips":["Full EDA report","Outlier detection","Feature importance",
                          "Data cleaning","Predictive model","Visualize all columns"]},
                {"id":"stats","icon":"📈","label":"R / Statistics",
                 "color":"#06d6a0","bg":"rgba(6,214,160,.1)",
                 "desc":"Regression · ANOVA · Survival analysis · Publication-quality output",
                 "chips":["Linear regression","Descriptive statistics","Correlation matrix",
                          "One-way ANOVA","Logistic regression","Kaplan-Meier survival"]},
                {"id":"code","icon":"💻","label":"Code Executor",
                 "color":"#a855f7","bg":"rgba(168,85,247,.1)",
                 "desc":"Python · R · Julia · MATLAB · Upload PDF exercises → solved automatically",
                 "chips":["Solve PDF exercises","Run uploaded code","Execute and explain",
                          "Multi-file analysis","Debug code","Generate report"]},
                {"id":"engineering","icon":"🏗","label":"Engineering",
                 "color":"#fb923c","bg":"rgba(251,146,60,.1)",
                 "desc":"Structural · Chemical · Electrical · Fluid · Eurocode · Aspen level",
                 "chips":["Beam deflection 6m 50kN","Distillation column design",
                          "RLC circuit analysis","Pipe flow calculation",
                          "Column buckling","Heat exchanger LMTD"]},
            ]

            # ── 30 COMING SOON TOOLS ──────────────────────────────────────
            COMING = [
                {"id":"finance","icon":"💹","label":"Finance / Bloomberg","color":"#f72585"},
                {"id":"sql","icon":"🗄","label":"SQL Intelligence","color":"#00d4ff"},
                {"id":"ml","icon":"🧬","label":"Machine Learning","color":"#a855f7"},
                {"id":"matlab","icon":"📐","label":"MATLAB / Signal","color":"#60a5fa"},
                {"id":"math","icon":"🧮","label":"Mathematica","color":"#e879f9"},
                {"id":"chem","icon":"🧪","label":"ChemCAD / Aspen","color":"#34d399"},
                {"id":"physics","icon":"⚛","label":"Physics","color":"#fbbf24"},
                {"id":"bio","icon":"🔬","label":"Bioinformatics","color":"#4ade80"},
                {"id":"gis","icon":"🌍","label":"GIS / ArcGIS","color":"#22d3ee"},
                {"id":"excel","icon":"📉","label":"Excel / Sheets","color":"#16a34a"},
                {"id":"autocad","icon":"✏","label":"AutoCAD 2D","color":"#fb923c"},
                {"id":"elec","icon":"⚡","label":"Electrical / PSpice","color":"#fde047"},
                {"id":"fluid","icon":"🌊","label":"Fluid / CFD","color":"#38bdf8"},
                {"id":"env","icon":"🌱","label":"Environmental","color":"#86efac"},
                {"id":"econ","icon":"📊","label":"Econometrics / Stata","color":"#c084fc"},
                {"id":"medical","icon":"💊","label":"Clinical Statistics","color":"#f87171"},
                {"id":"latex","icon":"📝","label":"LaTeX / Academic","color":"#94a3b8"},
                {"id":"ansys","icon":"🔩","label":"ANSYS / FEA","color":"#f97316"},
                {"id":"energy","icon":"☀","label":"Energy / HOMER","color":"#facc15"},
                {"id":"image","icon":"📸","label":"Image Analysis","color":"#a78bfa"},
                {"id":"nlp","icon":"🔤","label":"NLP / Text Analysis","color":"#34d399"},
                {"id":"network","icon":"🌐","label":"Network Analysis","color":"#60a5fa"},
                {"id":"scraping","icon":"🕷","label":"Web Scraping","color":"#fb7185"},
                {"id":"neuro","icon":"🧠","label":"Neuroscience","color":"#c4b5fd"},
                {"id":"astro","icon":"🔭","label":"Astronomy","color":"#818cf8"},
                {"id":"drug","icon":"💉","label":"Drug Discovery","color":"#86efac"},
                {"id":"project","icon":"📋","label":"Project / Gantt","color":"#fbbf24"},
                {"id":"quality","icon":"🏭","label":"Six Sigma / Quality","color":"#fb923c"},
                {"id":"actuarial","icon":"🏦","label":"Actuarial","color":"#a5b4fc"},
                {"id":"present","icon":"🎬","label":"Cinematic Presentation","color":"#f72585"},
            ]

            # Active tool card
            sel = active_tool or "powerbi"
            active_cfg = next((t for t in ACTIVE if t["id"]==sel), ACTIVE[0])

            # Active tool cards row
            active_cards = html.Div(style={
                "display":"flex","gap":"8px","flexWrap":"wrap","marginBottom":"16px"},
            children=[
                html.Div(
                    id={"type":"tool-card","index":t["id"]},
                    n_clicks=0,
                    style={
                        "padding":"10px 14px","borderRadius":"12px","cursor":"pointer",
                        "border":"2px solid "+("" + t["color"] if t["id"]==sel else "rgba(255,255,255,.08)"),
                        "background": t["bg"] if t["id"]==sel else "rgba(255,255,255,.03)",
                        "minWidth":"140px","flex":"1","maxWidth":"200px",
                        "transition":"all .2s",
                    },
                    children=[
                        html.Div(style={"display":"flex","alignItems":"center","gap":"6px","marginBottom":"4px"},children=[
                            html.Span(t["icon"],style={"fontSize":"18px"}),
                            html.Span(t["label"],style={"fontWeight":"700","color":t["color"],"fontSize":"13px"}),
                        ]),
                        html.Div(t["desc"][:60]+"...",style={"fontSize":"10px","color":MUT,"lineHeight":"1.3"}),
                    ]
                ) for t in ACTIVE
            ])

            # Coming soon section
            coming_pills = html.Div(style={
                "display":"flex","flexWrap":"wrap","gap":"6px","marginTop":"8px"},
            children=[
                html.Div(style={
                    "padding":"5px 10px","borderRadius":"20px","fontSize":"11px",
                    "color":t["color"],"border":"1px solid "+t["color"],
                    "opacity":"0.55","cursor":"default",
                    "display":"flex","alignItems":"center","gap":"4px"},
                children=[
                    html.Span(t["icon"]),
                    html.Span(t["label"]),
                    html.Span("Soon",style={"fontSize":"9px","background":"rgba(255,255,255,.08)",
                              "borderRadius":"4px","padding":"1px 4px","marginLeft":"2px"}),
                ]) for t in COMING
            ])

            # Smart suggestion chips based on active tool + data
            tool_chips = active_cfg.get("chips",[])
            chip_row = html.Div(
                id="sugg-row",
                style={"display":"flex","flexWrap":"wrap","gap":"5px","marginBottom":"10px"},
                children=[
                    html.Span("Try: ",style={"fontSize":"11px","color":MUT,"alignSelf":"center",
                              "fontWeight":"600","marginRight":"2px","flexShrink":"0"}),
                ] + [
                    html.Button(c,id={"type":"chip","index":i},n_clicks=0,
                        style={"background":"rgba(67,97,238,.1)","color":ACC2,
                               "border":"1px solid rgba(76,201,240,.2)","borderRadius":"20px",
                               "padding":"4px 12px","fontSize":"11px","cursor":"pointer",
                               "fontWeight":"500","whiteSpace":"nowrap"})
                    for i,c in enumerate(tool_chips)
                ]
            )

            return html.Div([
                # ── TOOL SELECTOR ─────────────────────────────────────────
                html.Div(style={"padding":"16px 0 8px 0"},children=[
                    html.Div(style={"display":"flex","justifyContent":"space-between",
                                    "alignItems":"center","marginBottom":"10px"},children=[
                        html.Div([
                            html.Span("🔧 ",style={"fontSize":"14px"}),
                            html.Span("Select Tool",style={"fontWeight":"700","color":TXT,"fontSize":"14px"}),
                            html.Span(" — 5 active · 30 coming soon",
                                style={"color":MUT,"fontSize":"11px","marginLeft":"8px"}),
                        ]),
                        html.Span("⚡ Powered by Claude + DeepSeek",
                            style={"fontSize":"10px","color":MUT,"background":"rgba(67,97,238,.1)",
                                   "padding":"3px 8px","borderRadius":"20px","border":"1px solid rgba(67,97,238,.2)"}),
                    ]),
                    active_cards,

                    # Coming soon collapsible
                    html.Details(style={"marginTop":"4px"},children=[
                        html.Summary(style={
                            "cursor":"pointer","fontSize":"11px","color":MUT,
                            "padding":"6px 0","userSelect":"none"},
                            children="🔜 30 more tools coming soon — click to see"),
                        html.Div(style={
                            "background":"rgba(255,255,255,.02)","borderRadius":"10px",
                            "padding":"12px","border":"1px solid "+BDR,"marginTop":"6px"},
                        children=[
                            html.Div("Join waitlist to be notified when your tool is ready:",
                                style={"fontSize":"11px","color":MUT,"marginBottom":"8px"}),
                            coming_pills,
                            html.Div(style={"display":"flex","gap":"8px","marginTop":"10px"},children=[
                                dcc.Input(id="waitlist-email",type="email",
                                    placeholder="your@email.com — get notified first",
                                    style=inp({"flex":"1","height":"32px","fontSize":"12px"})),
                                html.Button("Join Waitlist",id="waitlist-btn",n_clicks=0,
                                    style=btn(extra={"padding":"6px 14px","fontSize":"12px","height":"32px"})),
                            ]),
                            html.Div(id="waitlist-status",style={"fontSize":"11px","color":SUC,"marginTop":"4px"}),
                        ])
                    ]),
                ]),

                # ── UPLOAD SECTION ────────────────────────────────────────
                html.Div(style=card(),children=[
                    html.Div(style={"display":"flex","justifyContent":"space-between",
                                    "alignItems":"center","marginBottom":"10px"},children=[
                        h2("📤 Upload Data or Files"),
                        html.Div([
                            html.Span("Any file type accepted",style={
                                "fontSize":"11px","color":MUT,"background":"rgba(67,97,238,.08)",
                                "padding":"3px 8px","borderRadius":"20px",
                                "border":"1px solid rgba(67,97,238,.2)"}),
                        ]),
                    ]),
                    html.Div(style={"fontSize":"11px","color":MUT,"marginBottom":"8px"},
                        children=[
                            html.Span("📄 CSV · Excel · JSON · PDF · Images · "),
                            html.Span(".py · .R · .jl · .m · .ipynb · ",style={"color":ACC2}),
                            html.Span("DXF · Audio · Video · Word · Any format"),
                        ]),
                    dcc.Loading(type="circle",color=PRI,children=[
                        dcc.Upload(id="upload-file",multiple=True,max_size=500*1024*1024,
                            children=html.Div([
                                html.Div("⬆",style={"fontSize":"36px","color":PRI,"marginBottom":"6px"}),
                                html.Div("Click or drag & drop any file",
                                    style={"color":MUT,"fontSize":"14px","fontWeight":"600"}),
                                html.Div("CSV, Excel, PDF, images, code files (.py .R .jl), DXF, audio, video...",
                                    style={"fontSize":"11px","color":"#444","marginTop":"3px"}),
                            ],style={"textAlign":"center","padding":"22px"}),
                            style={"border":"2px dashed "+PRI,"borderRadius":"12px",
                                   "cursor":"pointer","background":"rgba(67,97,238,.03)",
                                   "marginBottom":"8px"}),
                        html.Div(id="upload-status",style={"fontSize":"13px","fontWeight":"600"}),
                    ]),
                ]),

                html.Div(id="preview-area"),
                html.Div(id="auto-area"),

                # ── ANALYZE SECTION ───────────────────────────────────────
                html.Div(id="analyze-section",style={"display":"block"},children=[
                    html.Div(style=card(),children=[
                        # Selected tool header
                        html.Div(style={
                            "display":"flex","alignItems":"center","gap":"8px","marginBottom":"12px",
                            "padding":"8px 12px","borderRadius":"8px",
                            "background":active_cfg["bg"],
                            "border":"1px solid "+active_cfg["color"].replace(")",",0.3)").replace("rgb","rgba") if active_cfg["color"].startswith("rgb") else active_cfg["color"]+"44"},
                        children=[
                            html.Span(active_cfg["icon"],style={"fontSize":"20px"}),
                            html.Div([
                                html.Div(active_cfg["label"],style={"fontWeight":"700","color":active_cfg["color"],"fontSize":"14px"}),
                                html.Div(active_cfg["desc"],style={"fontSize":"11px","color":MUT}),
                            ]),
                        ]),

                        # Smart suggestion chips
                        chip_row,

                        # Query input
                        html.Div(style={"display":"flex","gap":"8px","marginBottom":"10px"},children=[
                            dcc.Input(id="qbox",type="text",debounce=False,n_submit=0,
                                placeholder="Ask anything, describe what you want, or just click Run...",
                                style=inp({"flex":"1"})),
                            html.Button("🎤",id="qmic",n_clicks=0,
                                style=btn(BG,PRI,{"border":"1.5px solid "+PRI,"width":"42px","padding":"0","flexShrink":"0"})),
                            html.Button("▶ Run",id="run-q",n_clicks=0,
                                style=btn(extra={"flexShrink":"0","padding":"10px 20px","fontWeight":"700"})),
                            html.Button("🔬 Deep",id="deep-btn",n_clicks=0,
                                style=btn(BG,"#7c3aed",{"border":"1.5px solid #7c3aed","flexShrink":"0"})),
                            html.Button("🧪 Science",id="sci-btn",n_clicks=0,
                                style=btn(BG,ACC2,{"border":"1.5px solid "+ACC2,"flexShrink":"0"})),
                        ]),

                        dcc.Loading(type="circle",color=PRI,
                            children=html.Div(id="query-out",style={"marginTop":"8px"})),
                        dcc.Loading(type="circle",color="#7c3aed",
                            children=html.Div(id="deep-out")),
                        dcc.Loading(type="circle",color=ACC2,
                            children=html.Div(id="sci-out")),
                    ]),
                ]),
            ])

        if tab=="chat":
            bubbles=[]
            for h_item in (chat_hist or []):
                is_ai=h_item["role"]=="assistant"
                bubbles.append(html.Div([
                    html.Span("🤖 " if is_ai else "👤 ",style={"flexShrink":"0","fontSize":"16px"}),
                    dcc.Markdown(h_item["content"],style={"fontSize":"13px","lineHeight":"1.6","color":TXT,"flex":"1"}),
                ],style={"display":"flex","gap":"10px","alignItems":"flex-start","padding":"12px 16px",
                         "borderRadius":"14px","marginBottom":"10px",
                         "background":"rgba(67,97,238,.1)" if is_ai else "rgba(6,214,160,.08)",
                         "border":"1px solid "+("rgba(67,97,238,.2)" if is_ai else "rgba(6,214,160,.15)"),
                         "maxWidth":"82%","marginLeft":"0" if is_ai else "auto"}))
            return html.Div(style=card(),children=[
                html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"center","marginBottom":"8px"},children=[
                    h2("💬 Chat with Zynetrax AI"),
                    html.Span("Powered by DeepSeek V3",style={"fontSize":"11px","color":MUT,"background":"rgba(255,255,255,.05)","padding":"3px 10px","borderRadius":"20px"}),
                ]),
                html.P("Ask anything — data, code, science, business, general knowledge...",style={"color":MUT,"fontSize":"13px","marginBottom":"14px"}),
                html.Div(id="chat-window",
                    style={"minHeight":"200px","maxHeight":"460px","overflowY":"auto","padding":"12px",
                           "background":BG,"borderRadius":"10px","border":"1px solid "+BDR,"marginBottom":"12px"},
                    children=bubbles or [html.Div("Start a conversation 👇",style={"color":MUT,"fontSize":"13px"})]),
                html.Div(style={"display":"flex","gap":"8px"},children=[
                    dcc.Input(id="chat-q",type="text",debounce=False,n_submit=0,
                        placeholder="Ask anything...",style=inp({"flex":"1"})),
                    html.Button("🎤",id="chat-mic",n_clicks=0,style=btn(BG,PRI,{"border":"1.5px solid "+PRI,"width":"44px","padding":"0","flexShrink":"0"})),
                    html.Button("Send",id="chat-send",n_clicks=0,style=btn(extra={"flexShrink":"0"})),
                ]),
                dcc.Loading(type="dot",color=PRI,children=html.Div(id="chat-status")),
            ])
        return html.Div("Tab not found",style={"color":DNG})

    # ── SEARCH ────────────────────────────────────────────────────────────────
    @app.callback(Output("sq","value"),
                  Input({"type":"qtag","index":ALL},"n_clicks"),
                  State({"type":"qtag","index":ALL},"children"),prevent_initial_call=True)
    def qtag(clicks,labels):
        t=ctx.triggered_id
        if not t or not any(c for c in (clicks or []) if c): raise PreventUpdate
        if isinstance(t,dict):
            idx=t.get("index",0)
            if idx<len(labels): return labels[idx]
        raise PreventUpdate

    @app.callback(Output("sout","children"),Output("page-n","data"),
                  Input("sbtn","n_clicks"),Input("snext","n_clicks"),Input("sq","n_submit"),
                  Input({"type":"qtag","index":ALL},"n_clicks"),
                  State("sq","value"),State("stype","value"),State("page-n","data"),
                  State({"type":"qtag","index":ALL},"children"),prevent_initial_call=True)
    def do_search(n1,n2,n3,qtags,query,ftype,page,labels):
        t=ctx.triggered_id
        if isinstance(t,dict) and t.get("type")=="qtag":
            idx=t.get("index",0)
            if qtags and any(qtags) and idx<len(labels): query=labels[idx]
        if not query or not query.strip(): return html.Div("Enter a search term.",style={"color":MUT}), page
        page=page+1 if t=="snext" else 1
        try: results=se.search(query.strip(),page,file_type=ftype or "ALL")
        except Exception: results=[]
        if not results:
            return html.Div([html.Div('No results for "{}".'.format(query),style={"color":MUT,"marginBottom":"10px"}),
                html.A(html.Button("Search Google →",style=btn()),href="https://www.google.com/search?q={}+dataset".format(query.replace(" ","+")),target="_blank")]), page
        TYPE_C={"CSV":("#d1fae5","#065f46"),"Excel":("#dbeafe","#1e40af"),"JSON":("#fef3c7","#92400e"),
                "PDF":("#fee2e2","#991b1b"),"Web":("#1e1e35","#8888aa"),"Image":("#ede9fe","#5b21b6"),"Video":("#fff7ed","#c2410c")}
        cards=[html.Div('"{}" — page {} — {} results'.format(query,page,len(results)),
                style={"padding":"8px 12px","background":"rgba(67,97,238,.08)","borderRadius":"8px","marginBottom":"12px","fontSize":"13px","color":PRI,"fontWeight":"600"})]
        for r in results:
            bg,fg=TYPE_C.get(r.get("type","Web"),("#1e1e35","#8888aa"))
            cards.append(html.Div(style={"background":CARD,"borderRadius":"10px","padding":"14px 18px","marginBottom":"10px","border":"1px solid "+BDR},children=[
                html.Div(style={"display":"flex","gap":"10px"},children=[
                    html.Span(r.get("icon","📄"),style={"fontSize":"22px","flexShrink":"0"}),
                    html.Div([
                        html.Div([html.Span(r["title"][:80],style={"fontWeight":"600","color":TXT,"fontSize":"14px"}),
                            html.Span(r.get("type",""),style={"display":"inline-block","padding":"2px 8px","borderRadius":"10px","fontSize":"10px","fontWeight":"700","background":bg,"color":fg,"marginLeft":"8px"}),
                            html.Span(r.get("sector",""),style={"display":"inline-block","padding":"2px 8px","borderRadius":"10px","fontSize":"10px","background":"rgba(76,201,240,.1)","color":ACC2,"marginLeft":"4px"}) if r.get("sector") else None,
                        ],style={"marginBottom":"5px"}),
                        html.Div(r.get("summary","")[:180],style={"fontSize":"13px","color":MUT,"marginBottom":"8px"}),
                        html.A(html.Button("Open Dataset →",style=btn(extra={"fontSize":"12px","padding":"5px 12px","height":"28px"})),href=r["url"],target="_blank"),
                    ],style={"flex":"1"}),
                ]),
            ]))
        return html.Div(cards), page

    # ── UPLOAD ────────────────────────────────────────────────────────────────
    @app.callback(
        Output("df-store","data"),Output("upload-status","children"),
        Output("preview-area","children"),Output("auto-area","children"),
        Output("analyze-section","style"),Output("sugg-row","children",allow_duplicate=True),
        Input("upload-file","contents"),State("upload-file","filename"),State("lang","data"),
        prevent_initial_call=True)
    def do_upload(contents,filenames,lang):
        if not contents: raise PreventUpdate
        lang=lang or "en"; ln=LN.get(lang,"English")
        # Show immediate processing feedback in upload-status
        dfs=[]; media_preview=None
        for content,fname in zip(contents,filenames or [""]*len(contents)):
            try:
                _,cs=content.split(",",1); raw=base64.b64decode(cs)
                ext=(fname or "").lower().rsplit(".",1)[-1]
                if ext in ("png","jpg","jpeg","gif","webp","bmp"):
                    try:
                        resp=_oai.chat.completions.create(model="gpt-4o-mini",max_tokens=500,
                            messages=[{"role":"user","content":[
                                {"type":"image_url","image_url":{"url":"data:image/{};base64,{}".format(ext,cs),"detail":"low"}},
                                {"type":"text","text":"Analyse this image in {}. What is it? Location? Key data?".format(ln)}]}])
                        analysis=resp.choices[0].message.content
                    except Exception: analysis="Image uploaded."
                    media_preview=html.Div(style=card(),children=[
                        html.Div("🖼 Image Analysis",style={"fontWeight":"700","color":ACC2,"fontSize":"16px","marginBottom":"12px"}),
                        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"},children=[
                            html.Img(src="data:image/{};base64,{}".format(ext,cs),style={"maxWidth":"300px","maxHeight":"260px","borderRadius":"10px","border":"1px solid "+BDR}),
                            dcc.Markdown(analysis,style={"flex":"1","fontSize":"13px","color":TXT,"lineHeight":"1.6"}),
                        ]),
                    ])
                    return no_update, html.Div("✅ "+fname,style={"color":SUC}), media_preview, html.Div(), {"display":"none"}, []
                elif ext in ("mp4","avi","mov","mkv","webm"):
                    try:
                        import cv2, PIL.Image as PILImage
                        tmp=tempfile.NamedTemporaryFile(suffix="."+ext,delete=False,mode="wb")
                        tmp.write(raw); tmp.close()
                        cap=cv2.VideoCapture(tmp.name)
                        fps=cap.get(cv2.CAP_PROP_FPS) or 30
                        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        dur=total/fps; w=int(cap.get(3)); h_v=int(cap.get(4))
                        frames=[]; step=max(1,total//4)
                        for i in range(1,5):
                            cap.set(cv2.CAP_PROP_POS_FRAMES,i*step)
                            ok,frame=cap.read()
                            if not ok: continue
                            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                            pil=PILImage.fromarray(frame)
                            scale=min(512/max(w,h_v),1.0)
                            pil=pil.resize((int(w*scale),int(h_v*scale)),PILImage.LANCZOS)
                            buf2=io.BytesIO(); pil.save(buf2,format="JPEG",quality=72)
                            frames.append("data:image/jpeg;base64,"+base64.b64encode(buf2.getvalue()).decode())
                        cap.release(); os.unlink(tmp.name)
                        msgs=[{"role":"user","content":[{"type":"text","text":"Analyse this video in {}. What is shown? Location? Events?".format(ln)}]+[{"type":"image_url","image_url":{"url":f,"detail":"low"}} for f in frames]}]
                        resp=_oai.chat.completions.create(model="gpt-4o-mini",max_tokens=600,messages=msgs)
                        analysis=resp.choices[0].message.content
                    except ImportError: analysis="Install opencv-python Pillow for video analysis."
                    except Exception: analysis="Video uploaded."
                    media_preview=html.Div(style=card(),children=[
                        html.Div("🎥 Video Analysis — {:.1f}s · {}×{}".format(dur,w,h_v),style={"fontWeight":"700","color":ACC2,"fontSize":"16px","marginBottom":"10px"}),
                        dcc.Markdown(analysis,style={"fontSize":"13px","color":TXT,"lineHeight":"1.6"}),
                    ])
                    return no_update, html.Div("✅ "+fname,style={"color":SUC}), media_preview, html.Div(), {"display":"none"}, []
                elif ext in ("xlsx","xls"): df=_read_excel(io.BytesIO(raw))
                elif ext=="json":
                    d=json.loads(raw); df=pd.json_normalize(d if isinstance(d,list) else [d])
                elif ext=="pdf":
                    try:
                        import pdfplumber
                        tables=[]
                        with pdfplumber.open(io.BytesIO(raw)) as pdf:
                            text="\n".join(pg.extract_text() or "" for pg in pdf.pages[:5])
                            for pg in pdf.pages[:10]:
                                t=pg.extract_table()
                                if t and len(t)>1: tables.append(pd.DataFrame(t[1:],columns=t[0]))
                        if tables: df=tables[0]
                        else:
                            analysis=_call_llm("Analyse in {}: {}".format(LN.get(lang,"English"),text[:3000]),max_tokens=600)
                            media_preview=html.Div(style=card(),children=[
                                html.Div("📄 PDF: "+fname,style={"fontWeight":"700","color":ACC2,"fontSize":"16px","marginBottom":"10px"}),
                                dcc.Markdown(analysis or "No content.",style={"fontSize":"13px","color":TXT}),
                            ])
                            return no_update, html.Div("✅ "+fname,style={"color":SUC}), media_preview, html.Div(), {"display":"none"}, []
                    except Exception: continue
                else: df=_read_csv(raw.decode("utf-8","replace"))
                dfs.append(df)
            except Exception as e: print("[Upload]",fname,e)
        if not dfs: return no_update, html.Div([
            html.Span("⚠ ",style={"color":WARN,"fontSize":"16px"}),
            html.Span("Could not read file. Please use CSV, Excel, or JSON.",
                style={"color":WARN,"fontSize":"13px","fontWeight":"600"})
        ],style={"padding":"8px 12px","background":"rgba(255,209,102,.08)","borderRadius":"8px",
                 "border":"1px solid rgba(255,209,102,.2)"}), html.Div(), html.Div(), {"display":"none"}, []

        try:
            engine=DataEngine(dfs if len(dfs)>1 else dfs[0]); engine.normalize(); engine._lang=lang
            clean=engine.df
        except Exception as eng_err:
            return no_update, html.Div([
                html.Span("❌ Processing error: ",style={"color":DNG,"fontWeight":"700"}),
                html.Span(str(eng_err)[:200],style={"color":MUT,"fontSize":"12px"})
            ]), html.Div(), html.Div(), {"display":"none"}, []
        
        for c in clean.columns:
            if pd.api.types.is_datetime64_any_dtype(clean[c]):
                clean[c]=clean[c].dt.strftime("%Y-%m-%d")
        qr=engine.quality_report(); s=engine._schema
        ROLES={"measure":("rgba(67,97,238,.15)",ACC2,"📐"),"dimension":("rgba(6,214,160,.12)",SUC,"🏷"),
               "temporal":("rgba(247,37,133,.12)",ACC,"📅"),"entity":("rgba(255,209,102,.12)",WARN,"🏢"),
               "year_col":("rgba(247,37,133,.12)",ACC,"📅"),"binary":("rgba(6,214,160,.12)",SUC,"🔘")}
        stats_row=html.Div([
            html.Div([html.Div(str(v),style={"fontSize":"24px","fontWeight":"800","color":c}),
                      html.Div(l,style={"fontSize":"11px","color":MUT,"marginTop":"2px"})],
                style={"background":CARD,"borderRadius":"10px","padding":"12px 16px","border":"1px solid "+BDR,"textAlign":"center","minWidth":"80px"})
            for v,l,c in [(qr["rows"],"Rows",PRI),(qr["cols"],"Cols",ACC2),(qr["measures"],"Measures",SUC),(qr["dimensions"],"Categories",WARN),(qr["temporals"],"Time",ACC)]
        ],style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"14px"})
        pills=html.Div([html.Span("{} {}".format(ROLES.get(m["role"],("","","?"))[2],c),
            style={"display":"inline-block","padding":"3px 10px","borderRadius":"20px","fontSize":"11px","fontWeight":"600",
                   "background":ROLES.get(m["role"],("rgba(255,255,255,.05)",MUT,""))[0],
                   "color":ROLES.get(m["role"],("",MUT,""))[1],"margin":"3px"})
            for c,m in s.items() if m["role"] not in ("id","text","binary")][:20],style={"marginBottom":"10px"})
        prev=clean.head(10)
        table=dash_table.DataTable(data=prev.to_dict("records"),columns=[{"name":c,"id":c} for c in prev.columns],
            style_table={"overflowX":"auto","borderRadius":"8px"},
            style_header={"backgroundColor":HDR,"color":"white","fontWeight":"700","fontSize":"12px","padding":"8px 12px","border":"none"},
            style_cell={"backgroundColor":CARD,"color":TXT,"fontSize":"12px","padding":"8px 12px","border":"1px solid "+BDR,"maxWidth":"160px","overflow":"hidden","textOverflow":"ellipsis"},
            style_data_conditional=[{"if":{"row_index":"odd"},"backgroundColor":"rgba(255,255,255,.02)"}],page_size=10)
        preview=html.Div(style=card(),children=[
            html.Div("📋 Data Preview",style={"fontSize":"16px","fontWeight":"700","color":TXT,"marginBottom":"12px"}),
            stats_row, pills, table,
            html.A(html.Button("⬇ Download Clean CSV",style={**btn(),"marginTop":"10px"}),
                   href="data:text/csv;charset=utf-8,"+clean.to_csv(index=False),download="clean.csv"),
        ])
        auto_figs=engine.get_auto_figs(); auto_ins=engine.get_auto_insight(lang=lang)
        auto_area=html.Div(style=card(),children=[
            html.Div("🤖 Auto Analysis",style={"fontSize":"16px","fontWeight":"700","color":TXT,"marginBottom":"4px"}),
            html.Div("Key insights generated automatically",style={"fontSize":"13px","color":MUT,"marginBottom":"14px"}),
            ibox(auto_ins), grid(auto_figs),
        ]) if auto_figs else html.Div()
        sugg=[html.Span("Try: ",style={"fontSize":"12px","color":MUT,"fontWeight":"600","alignSelf":"center","marginRight":"4px"}),
              *[html.Button(s,id={"type":"chip","index":i},n_clicks=0,
                  style=btn("rgba(67,97,238,.1)",ACC2,{"border":"1px solid rgba(76,201,240,.2)","fontSize":"11px","padding":"4px 10px","height":"26px","marginLeft":"4px"}))
                for i,s in enumerate(engine.get_suggestions(lang=lang))]]
        # Build file status badges
        # Build success status
        status_msg = html.Div([
            html.Span("✅ ",style={"color":SUC,"fontSize":"16px"}),
            html.Span(", ".join(filenames or []),
                style={"fontWeight":"700","color":TXT,"fontSize":"13px"}),
            html.Span("  —  {:,} rows × {} cols  ✓ Normalized".format(
                len(clean),len(clean.columns)),
                style={"color":MUT,"fontSize":"12px","marginLeft":"6px"}),
            html.Button("→ Analyze",id="goto-analyze-btn",n_clicks=0,
                style={**btn(extra={"padding":"4px 14px","fontSize":"12px","height":"28px",
                               "marginLeft":"12px","fontWeight":"700"})}),
        ],style={"display":"flex","alignItems":"center","flexWrap":"wrap","gap":"4px",
                 "padding":"8px 12px","background":"rgba(6,214,160,.08)",
                 "borderRadius":"8px","border":"1px solid rgba(6,214,160,.2)"})

        return clean.to_json(date_format="iso"), status_msg, preview, auto_area, {"display":"block"}, sugg

    @app.callback(Output("qbox","value"),
                  Input({"type":"chip","index":ALL},"n_clicks"),
                  State({"type":"chip","index":ALL},"children"),prevent_initial_call=True)
    def chip(clicks,labels):
        t=ctx.triggered_id
        if not t or not any(c for c in (clicks or []) if c): raise PreventUpdate
        if isinstance(t,dict):
            idx=t.get("index",0)
            if idx<len(labels): return labels[idx]
        raise PreventUpdate

    # ── TOOL SELECTOR — highlight active pill + update desc + update chips ──
    TOOL_CHIPS={
        "auto":  ["Top 10 by value","Trend over time","Correlation matrix","Distribution analysis","Build dashboard","Regression analysis"],
        "powerbi":["Build full dashboard","KPI scorecard","Stacked bar chart","Treemap","World map","Waterfall chart"],
        "sql":   ["SELECT * FROM data LIMIT 10","Top 10 by value","GROUP BY category COUNT","Filter WHERE value > 100","Aggregate SUM by group","JOIN on common column"],
        "stats": ["Linear regression","One-way ANOVA","Correlation matrix","Descriptive statistics","Kaplan-Meier survival","Distribution & normality test"],
        "ml":    ["K-means clustering","PCA analysis","Random Forest classification","Feature importance","Anomaly detection","Predict target variable"],
        "finance":["NPV and IRR analysis","Black-Scholes option pricing","Markowitz portfolio optimization","Stock price AAPL MSFT","DCF valuation","Risk metrics Sharpe ratio"],
        "struct":["Beam deflection 6m span 50kN UDL","Column buckling L=4m","Foundation bearing capacity B=1.5m","Cantilever beam 8m 30kN","Simply supported beam 10m 20kN","Steel section W200x52"],
        "chem":  ["McCabe-Thiele distillation xF=0.4 xD=0.95","Heat exchanger LMTD 150C to 80C","Ideal gas T=25 P=1 n=1","Vapor pressure ethanol","CSTR reactor design","Molecular weight H2SO4"],
        "matlab":["FFT frequency analysis","Butterworth lowpass filter fc=100Hz","Bode plot wn=10 zeta=0.5","PID step response","Signal processing 50Hz signal","Control system stability margins"],
        "math":  ["Derivative of x^3 + 2x^2 + sin(x)","Integrate x^2 + 3x dx","Solve x^2 - 5x + 6 = 0","Matrix determinant [[1,2],[3,4]]","Taylor series e^x","ODE dy/dx = 2y"],
        "physics":["Projectile v=20m/s angle=45","Particle in box n=5 L=1nm","Carnot engine 800K 300K","Snell refraction n1=1 n2=1.5","Wave f=440Hz v=343","de Broglie wavelength electron"],
        "bio":   ["DNA sequence ATGATCGATCGATCG","IC50 dose response EC50=1","ROC curve AUC analysis","Kaplan-Meier survival groups","GC content DNA analysis","Hill equation n=1.5"],
        "env":   ["Carbon footprint 1000kWh 200L fuel","Water quality BOD=250mg/L","CO2 emissions calculator","Dissolved oxygen stream","Greenhouse gas breakdown","Environmental impact assessment"],
        "elec":  ["RLC circuit R=1000 L=0.001 C=1e-6","Power factor pf=0.85 P=1000kW","3-phase 11kV load analysis","Resonant frequency calculation","Step response underdamped","Impedance vs frequency"],
        "fluid": ["Pipe flow D=0.1m L=100m v=2m/s","Moody diagram turbulent Re=50000","Reynolds number water pipe","Pressure drop calculation","Pump head flow rate","Darcy Weisbach friction"],
        "code":  ["Python: import pandas as pd","Run R regression analysis","SQL SELECT with GROUP BY","Python matplotlib plot","C++ hello world","Julia matrix operations"],


    }
    TOOL_DESCS={
        "auto":   "⚡ AI selects the best tool automatically based on your query and data",
        "powerbi":"📊 Power BI / Looker Studio — interactive dashboards, KPIs, maps, treemaps, waterfalls",
        "sql":    "🗄 SQL Engine (DuckDB) — write real SQL or ask in plain English — as powerful as SQL Server",
        "stats":  "📈 R / SPSS / SAS level statistics — regression, ANOVA, survival analysis, factor analysis",
        "ml":     "🧬 Machine Learning — clustering, PCA, Random Forest, anomaly detection — DataRobot level",
        "finance":"💹 Bloomberg Terminal equivalent — NPV, IRR, Black-Scholes Greeks, Markowitz portfolio",
        "struct": "🏗 AutoCAD / SAP2000 / STAAD.Pro — beam analysis, column buckling, foundation design, DXF output",
        "chem":   "🧪 ChemCAD / Aspen Plus — distillation (McCabe-Thiele), heat exchangers, reactor design",
        "matlab": "📐 MATLAB / Simulink — FFT, digital filters, Bode plots, PID control, step response",
        "math":   "🧮 Mathematica / Maple — symbolic derivatives, integrals, ODEs, matrix algebra",
        "physics":"⚛ Physics — projectile motion, quantum mechanics, thermodynamics, optics, waves",
        "bio":    "🔬 BioPython / GraphPad Prism — DNA analysis, dose-response, ROC curves, survival",
        "env":    "🌱 Environmental — carbon footprint, water quality (DO sag), EPA model calculations",
        "elec":   "⚡ PSpice / LTspice — RLC circuits, frequency response, power factor, 3-phase power",
        "fluid":  "🌊 PIPE-FLO / AFT Fathom — Darcy-Weisbach, Moody diagram, velocity profiles",
        "code":   "💻 Code Executor — run Python, R, SQL, C++, Julia with full library support",
    }

    @app.callback(
        Output("active-tool","data"),
        Input({"type":"tool-card","index":ALL},"n_clicks"),
        State({"type":"tool-card","index":ALL},"id"),
        prevent_initial_call=True)
    def select_tool(clicks, ids):
        t = ctx.triggered_id
        if not t or not any(c for c in (clicks or []) if c):
            raise PreventUpdate
        tool_id = t.get("index","powerbi") if isinstance(t,dict) else "powerbi"
        return tool_id

    @app.callback(
        Output("waitlist-status","children"),
        Input("waitlist-btn","n_clicks"),
        State("waitlist-email","value"),
        prevent_initial_call=True)
    def join_waitlist(n, email):
        if not n: raise PreventUpdate
        if not email or "@" not in email:
            return html.Span("⚠ Please enter a valid email",style={"color":WARN})
        return html.Span("✅ You're on the waitlist! We'll notify you when new tools launch.",style={"color":SUC})

    @app.callback(
        Output("sugg-row","children",allow_duplicate=True),
        Input("active-tool","data"),
        State("df-store","data"),
        State("lang","data"),
        prevent_initial_call=True)
    def update_chips(tool, df_data, lang):
        tool = tool or "auto"
        lang = lang or "en"
        chips_list = TOOL_CHIPS.get(tool, TOOL_CHIPS["auto"])
        if tool == "auto" and df_data:
            try:
                df = pd.read_json(io.StringIO(df_data))
                engine = DataEngine(df)
                engine._schema = fingerprint_schema(df)
                chips_list = engine.get_suggestions(lang)
            except Exception:
                pass
        return [html.Span("Try: ",style={"fontSize":"12px","color":MUT,"fontWeight":"600",
                          "alignSelf":"center","marginRight":"4px","flexShrink":"0"})] + [
            html.Button(c, id={"type":"chip","index":i}, n_clicks=0,
                style={"background":"rgba(67,97,238,.12)","color":ACC2,
                       "border":"1px solid rgba(76,201,240,.2)","borderRadius":"20px",
                       "padding":"4px 12px","fontSize":"11px","cursor":"pointer",
                       "fontWeight":"500","whiteSpace":"nowrap"})
            for i,c in enumerate(chips_list)]

    # ── SMART QUERY ROUTING — use active tool to route ──────────────────────

    @app.callback(
        Output("tool-desc-display","children"),
        Input("active-tool","data"),
        prevent_initial_call=True)
    def sync_tool_desc(tool):
        TOOL_DESCS2={
            "auto":   "⚡ AI selects the best tool automatically",
            "powerbi":"📊 Power BI — dashboards, KPIs, maps, treemaps",
            "sql":    "🗄 SQL Engine — write SQL or plain English — DuckDB",
            "stats":  "📈 R/SPSS/SAS statistics — regression, ANOVA, survival",
            "ml":     "🧬 Machine Learning — clustering, PCA, Random Forest",
            "finance":"💹 Bloomberg — NPV, Black-Scholes, Markowitz portfolio",
            "struct": "🏗 AutoCAD/SAP2000 — beam, column, foundation, DXF",
            "chem":   "🧪 ChemCAD/Aspen — distillation, heat exchanger, reactor",
            "matlab": "📐 MATLAB — FFT, filters, Bode, PID control",
            "math":   "🧮 Mathematica — derivatives, integrals, ODEs, matrices",
            "physics":"⚛ Physics — projectile, quantum, thermodynamics, optics",
            "bio":    "🔬 BioPython — DNA analysis, dose-response, ROC curves",
            "env":    "🌱 Environmental — carbon footprint, water quality",
            "elec":   "⚡ PSpice/LTspice — RLC circuits, power factor, 3-phase",
            "fluid":  "🌊 PIPE-FLO — Darcy-Weisbach, Moody, velocity profiles",
            "code":   "💻 Code Executor — Python, R, SQL, C++, Julia",
        }
        return TOOL_DESCS2.get(tool or "auto","")

    @app.callback(Output("query-out","children"),Output("figs-store","data"),Output("insight-store","data"),
                  Input("run-q","n_clicks"),Input("qbox","n_submit"),
                  State("qbox","value"),State("df-store","data"),State("lang","data"),
                  State("active-tool","data"),
                  prevent_initial_call=True)
    def run_query(n,ns,query,df_data,lang,active_tool):
        if not query or not query.strip():
            return html.Div("Enter a question.",style={"color":MUT}), [], ""
        lang=lang or "en"; active_tool=active_tool or "auto"

        # Engineering/science tools work WITHOUT data
        no_data_tools=["struct","chem","matlab","math","physics","env","elec","fluid","finance"]
        if not df_data and active_tool not in no_data_tools:
            # Try answering with LLM even without data for general queries
            if not df_data:
                ln={"en":"English","de":"German","fr":"French","es":"Spanish","ur":"Urdu","ar":"Arabic"}.get(lang,"English")
                ans=_call_llm("Answer in {} as expert analyst: {}".format(ln,query),max_tokens=600)
                return html.Div([ibox(ans)]), [], ans

        lang_name={"en":"English","de":"German","fr":"French","es":"Spanish","ur":"Urdu",
                   "ar":"Arabic","zh":"Chinese","hi":"Hindi","tr":"Turkish","ru":"Russian"}.get(lang,"English")
        figs=[]; ins=""

        try:
            df = pd.read_json(io.StringIO(df_data)) if df_data else None

            # Route to specific tool if selected
            from intelligence.data_engine import (
                structural_engineering, chemical_engineering, advanced_statistics,
                machine_learning, financial_engineering, signal_processing,
                symbolic_math, biology, physics, fluid_dynamics,
                electrical_engineering, sql_engine, manufacturing_quality,
                econometrics, environmental_science, code_executor,
                build_powerbi_dashboard, fingerprint_schema, solve
            )

            if active_tool == "powerbi" and df is not None:
                schema = fingerprint_schema(df)
                # If user typed a specific query, use Claude to build that specific chart
                # If no specific query or generic "build dashboard", build full dashboard
                generic_queries = ["build full dashboard", "build dashboard", "dashboard", "kpi scorecard", ""]
                is_generic = not query or query.lower().strip() in generic_queries
                if is_generic:
                    figs = build_powerbi_dashboard(df, schema, lang)
                    ins = _call_llm("Summarize this dashboard data in {}: columns {} sample {}".format(
                        lang_name, list(df.columns[:6]), df.head(3).to_dict()), max_tokens=300)
                else:
                    # Specific query — use solve() which routes intelligently
                    dfs_dict = {"data": df}
                    res = solve(query, df, dfs_dict, lang)
                    figs = res.get("figs", [])
                    ins = res.get("insight", "")
                    results_text = res.get("results", [])
                    if results_text: ins = "\n".join(str(r) for r in results_text[:3]) + "\n\n" + ins
                    # If still no figs, fall back to full dashboard
                    if not figs:
                        figs = build_powerbi_dashboard(df, schema, lang)

            elif active_tool == "sql" and df is not None:
                res = sql_engine(query, {"data":df}, lang)
                figs=res.get("figs",[]); ins=res.get("insight","")
                results=res.get("results",[])
                if results: ins = "\n".join(str(r) for r in results) + "\n\n" + ins

            elif active_tool == "stats" and df is not None:
                res = advanced_statistics(df, query, lang)
                figs=res.get("figs",[]); ins=res.get("insight","")
                results=res.get("results",[])
                if results: ins = "\n".join(str(r) for r in results[:3]) + "\n\n" + ins

            elif active_tool == "ml" and df is not None:
                res = machine_learning(df, query, lang)
                figs=res.get("figs",[]); ins=res.get("insight","")

            elif active_tool == "finance":
                res = financial_engineering(query, df, lang)
                figs=res.get("figs",[]); ins=res.get("insight","")
                results=res.get("results",[])
                if results: ins = "\n".join(str(r) for r in results[:3]) + "\n\n" + ins

            elif active_tool == "struct":
                res = structural_engineering(query, df, lang)
                figs=res.get("figs",[]); ins=res.get("insight","")
                results=res.get("results",[])
                if results: ins = "\n".join(str(r) for r in results[:3]) + "\n\n" + ins

            elif active_tool == "chem":
                res = chemical_engineering(query, df, lang)
                figs=res.get("figs",[]); ins=res.get("insight","")
                results=res.get("results",[])
                if results: ins = "\n".join(str(r) for r in results[:3]) + "\n\n" + ins

            elif active_tool == "matlab":
                res = signal_processing(query, df, lang)
                figs=res.get("figs",[]); ins=res.get("insight","")
                results=res.get("results",[])
                if results: ins = "\n".join(str(r) for r in results[:3]) + "\n\n" + ins

            elif active_tool == "math":
                res = symbolic_math(query, lang)
                figs=res.get("figs",[]); ins=res.get("insight","")
                results=res.get("results",[])
                if results: ins = "\n".join(str(r) for r in results[:3]) + "\n\n" + ins

            elif active_tool == "physics":
                res = physics(query, lang)
                figs=res.get("figs",[]); ins=res.get("insight","")
                results=res.get("results",[])
                if results: ins = "\n".join(str(r) for r in results[:3]) + "\n\n" + ins

            elif active_tool == "bio":
                res = biology(query, df, lang)
                figs=res.get("figs",[]); ins=res.get("insight","")

            elif active_tool == "env":
                res = environmental_science(query, df, lang)
                figs=res.get("figs",[]); ins=res.get("insight","")
                results=res.get("results",[])
                if results: ins = "\n".join(str(r) for r in results[:3]) + "\n\n" + ins

            elif active_tool == "elec":
                res = electrical_engineering(query, lang)
                figs=res.get("figs",[]); ins=res.get("insight","")
                results=res.get("results",[])
                if results: ins = "\n".join(str(r) for r in results[:3]) + "\n\n" + ins

            elif active_tool == "fluid":
                res = fluid_dynamics(query, lang)
                figs=res.get("figs",[]); ins=res.get("insight","")
                results=res.get("results",[])
                if results: ins = "\n".join(str(r) for r in results[:3]) + "\n\n" + ins

            elif active_tool == "code":
                res = code_executor(query, df, lang)
                figs=res.get("figs",[]); ins=res.get("insight","")
                results=res.get("results",[])
                if results: ins = "\n".join(str(r) for r in results[:3])

            else:
                # AUTO mode — use solve() which routes intelligently
                # PLUS: if no chart returned, LLM generates one from data
                dfs_dict = {"data":df} if df is not None else {}
                res = solve(query, df, dfs_dict, lang)
                figs=res.get("figs",[]); ins=res.get("insight","")
                results=res.get("results",[])
                if results: ins = "\n".join(str(r) for r in results[:3]) + "\n\n" + ins

                # GUARANTEE CHART: If no figs returned but we have data, LLM builds chart
                if not figs and df is not None:
                    try:
                        schema = fingerprint_schema(df)
                        measures=[c for c,m in schema.items() if m["role"]=="measure"]
                        dims=[c for c,m in schema.items() if m["role"]=="dimension"]
                        entities=[c for c,m in schema.items() if m["role"]=="entity"]
                        # Ask LLM what chart to build
                        col_info = {c: {"type":schema[c]["role"],"sample":schema[c]["sample"][:3]}
                                   for c in list(df.columns)[:12]}
                        chart_prompt = (
                            "User asked: \'{}\'. "
                            "Data columns: {}. "
                            "Return JSON with keys: chart_type, x_col, y_col, color_col, agg, top_n, title, orientation. "
                            "chart_type: bar/line/scatter/pie/histogram/box. "
                            "agg: sum/mean/count/max/min. "
                            "Return ONLY the JSON object, nothing else."
                        ).format(query, json.dumps(col_info))
                        # Use Claude for precise JSON chart spec
                        chart_spec_raw = _call_claude_app(chart_prompt, max_tokens=300,
                            system="You are a data visualization expert. Return ONLY valid JSON, no markdown, no explanation.")
                        if not chart_spec_raw:
                            chart_spec_raw = _call_llm(chart_prompt, max_tokens=200,
                                system="Return only valid JSON.")
                        chart_spec_raw = re.sub(r"```json|```","",chart_spec_raw).strip()
                        chart_spec = json.loads(chart_spec_raw)
                        ct = chart_spec.get("chart_type","bar")
                        xc = chart_spec.get("x_col","")
                        yc = chart_spec.get("y_col","")
                        cc = chart_spec.get("color_col")
                        agg= chart_spec.get("agg","sum")
                        topn=int(chart_spec.get("top_n",15))
                        title=chart_spec.get("title","Analysis")
                        orient=chart_spec.get("orientation","v")

                        if xc in df.columns and yc in df.columns:
                            plot_df = df[[xc,yc]+(([cc] if cc and cc in df.columns else []))]
                            if agg in ["sum","mean","count","max","min"] and pd.api.types.is_numeric_dtype(df[yc]):
                                grp = plot_df.groupby(xc)[yc].agg(agg).reset_index()
                                grp = grp.nlargest(topn, yc) if topn else grp
                            else:
                                grp = plot_df.head(topn)

                            if ct=="bar":
                                if orient=="h":
                                    fig=px.bar(grp,x=yc,y=xc,orientation="h",
                                        color=yc,color_continuous_scale="Blues",title=title)
                                else:
                                    fig=px.bar(grp,x=xc,y=yc,color=yc,
                                        color_continuous_scale="Blues",title=title)
                                fig.update_coloraxes(showscale=False)
                            elif ct=="line":
                                fig=px.line(grp,x=xc,y=yc,title=title,
                                    color_discrete_sequence=["#4361ee"])
                            elif ct=="scatter":
                                fig=px.scatter(df.head(500),x=xc,y=yc,
                                    color=cc if cc and cc in df.columns else None,
                                    title=title,opacity=0.7,trendline="ols")
                            elif ct=="pie":
                                fig=px.pie(grp,names=xc,values=yc,hole=0.45,title=title)
                            elif ct=="histogram":
                                fig=px.histogram(df,x=xc,nbins=30,title=title,
                                    color_discrete_sequence=["#4361ee"])
                            elif ct=="box":
                                fig=px.box(df,x=xc if xc in df.columns else None,
                                    y=yc,color=cc if cc and cc in df.columns else None,title=title)
                            else:
                                fig=px.bar(grp,x=xc,y=yc,title=title)

                            fig.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.03)",
                                font=dict(color="#c8c8d8",size=12),margin=dict(l=50,r=20,t=50,b=50))
                            fig.update_xaxes(gridcolor="rgba(255,255,255,.06)",zeroline=False)
                            fig.update_yaxes(gridcolor="rgba(255,255,255,.06)",zeroline=False)
                            figs=[fig]
                    except Exception as chart_err:
                        # Final fallback: simple top-N bar from measures/entities
                        try:
                            schema2 = fingerprint_schema(df)
                            m=[c for c,v in schema2.items() if v["role"]=="measure"]
                            e=[c for c,v in schema2.items() if v["role"] in ("entity","dimension")]
                            if m and e:
                                g=df[[e[0],m[0]]].groupby(e[0])[m[0]].sum().nlargest(15).reset_index()
                                fig=px.bar(g,x=m[0],y=e[0],orientation="h",
                                    color=m[0],color_continuous_scale="Blues",
                                    title="📊 {}  by  {}".format(m[0],e[0]))
                                fig.update_coloraxes(showscale=False)
                                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.03)",
                                    font=dict(color="#c8c8d8"),margin=dict(l=50,r=20,t=50,b=50))
                                figs=[fig]
                        except Exception: pass

        except Exception as e:
            ins = _call_llm("Answer this query as expert in {}: {}".format(lang_name,query),max_tokens=700)

        # Build UI
        import plotly.io as pio
        ui=[]
        if ins: ui.append(ibox(ins))
        if figs: ui.append(grid(figs))
        elif not ins:
            ui.append(html.Div("No results generated.",style={"color":MUT}))

        try: figs_json=[pio.to_json(f) for f in figs]
        except Exception: figs_json=[]
        return html.Div(ui), figs_json, ins

    @app.callback(Output("deep-out","children"),Input("deep-btn","n_clicks"),
                  State("df-store","data"),State("lang","data"),prevent_initial_call=True)
    def deep(n,data,lang):
        if not data: return html.Div("Upload data first.",style={"color":WARN})
        try:
            df=pd.read_json(io.StringIO(data))
            schema=fingerprint_schema(df)
            res=advanced_analysis(df,schema,lang=lang or "en")
            figs=res.get("figs",[]); findings=res.get("findings",[]); note=res.get("llm_insight","")
            ui=[html.Div("🔬 Deep Analysis",style={"fontWeight":"700","color":"#7c3aed","fontSize":"15px","marginBottom":"10px"})]
            if findings: ui.append(ibox("\n\n".join(findings)))
            if note: ui.append(ibox("🧠 "+note))
            if figs: ui.append(grid(figs))
            return html.Div(ui)
        except Exception as e: return html.Div("Error: {}".format(str(e)[:120]),style={"color":DNG})

    @app.callback(Output("sci-out","children"),Input("sci-btn","n_clicks"),
                  State("qbox","value"),State("df-store","data"),State("lang","data"),prevent_initial_call=True)
    def science(n,query,data,lang):
        if not data: return html.Div("Upload data first.",style={"color":WARN})
        if not query: return html.Div("Enter a query first.",style={"color":WARN})
        try:
            df=pd.read_json(io.StringIO(data))
            schema=fingerprint_schema(df)
            res=scientific_analysis(df,schema,query,lang=lang or "en")
            figs=res.get("figs",[]); results=res.get("results",[]); ins=res.get("insight","")
            rtype=res.get("type","Science")
            ui=[html.Div("🧪 "+rtype,style={"fontWeight":"700","color":ACC2,"fontSize":"15px","marginBottom":"10px"})]
            if results: ui.append(ibox("\n\n".join(results)))
            if ins: ui.append(ibox("🧠 "+ins))
            if figs: ui.append(grid(figs))
            return html.Div(ui)
        except Exception as e: return html.Div("Error: {}".format(str(e)[:120]),style={"color":DNG})

    return app

def run_server():
    app = run_dashboard()
    app.run_server(debug=False, host="127.0.0.1", port=8050)

if __name__ == "__main__":
    run_server()