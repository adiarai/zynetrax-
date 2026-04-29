"""
backend/app_generator.py — Generate complete web apps from descriptions
"""
import base64
import zipfile
import io


TEMPLATES = {
    "webapp": """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0a0a14;color:#e8e8f0;font-family:'Segoe UI',Arial,sans-serif;min-height:100vh}}
.header{{background:#12121f;padding:16px 24px;border-bottom:1px solid #1e1e35;display:flex;align-items:center;justify-content:space-between}}
.logo{{font-size:20px;font-weight:800;color:#4361ee}}
.main{{max-width:1200px;margin:0 auto;padding:32px 20px}}
.card{{background:#12121f;border-radius:14px;padding:24px;margin-bottom:20px;border:1px solid #1e1e35}}
h1{{font-size:28px;font-weight:800;margin-bottom:8px}}
h2{{font-size:20px;font-weight:700;margin-bottom:16px;color:#4cc9f0}}
.btn{{background:#4361ee;color:white;border:none;border-radius:8px;padding:12px 24px;font-size:14px;font-weight:600;cursor:pointer}}
.btn:hover{{background:#3251d4}}
.btn-ghost{{background:transparent;color:#4361ee;border:1.5px solid #4361ee}}
input,textarea,select{{background:#1a1a2e;color:#e8e8f0;border:1.5px solid #1e1e35;border-radius:8px;padding:10px 14px;font-size:14px;width:100%;margin-bottom:12px;outline:none}}
input:focus,textarea:focus{{border-color:#4361ee}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}}
.tag{{display:inline-block;background:rgba(67,97,238,.15);color:#4cc9f0;border-radius:20px;padding:3px 12px;font-size:12px;margin:3px}}
</style>
</head>
<body>
<div class="header">
  <div class="logo">{title}</div>
  <button class="btn" onclick="alert('Welcome!')">Get Started</button>
</div>
<div class="main">
  <div class="card">
    <h1>{title}</h1>
    <p style="color:#8888aa;margin-bottom:20px">{description}</p>
    <div id="app-content">{content}</div>
  </div>
</div>
<script>
// App logic here
console.log('{title} loaded');
{js}
</script>
</body>
</html>""",
    "dashboard": """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><title>{title} Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0a0a14;color:#e8e8f0;font-family:'Segoe UI',Arial,sans-serif}}
.sidebar{{width:220px;background:#12121f;height:100vh;position:fixed;border-right:1px solid #1e1e35;padding:20px 16px}}
.main{{margin-left:220px;padding:24px}}
.kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}}
.kpi{{background:#12121f;border-radius:12px;padding:20px;border:1px solid #1e1e35;text-align:center}}
.kpi-val{{font-size:28px;font-weight:800;color:#4361ee}}
.kpi-lbl{{font-size:12px;color:#8888aa;margin-top:4px}}
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.chart-card{{background:#12121f;border-radius:12px;padding:20px;border:1px solid #1e1e35}}
.logo{{font-size:18px;font-weight:800;color:#4361ee;margin-bottom:24px}}
nav a{{display:block;padding:10px 12px;border-radius:8px;color:#8888aa;text-decoration:none;margin-bottom:4px;font-size:14px}}
nav a:hover,nav a.active{{background:rgba(67,97,238,.15);color:#4cc9f0}}
</style>
</head>
<body>
<div class="sidebar">
  <div class="logo">{title}</div>
  <nav>
    <a href="#" class="active">📊 Overview</a>
    <a href="#">📈 Analytics</a>
    <a href="#">👥 Users</a>
    <a href="#">⚙ Settings</a>
  </nav>
</div>
<div class="main">
  <h1 style="margin-bottom:20px">{title} Dashboard</h1>
  <div class="kpi-grid">
    {kpis}
  </div>
  <div class="chart-grid">
    <div class="chart-card"><h3 style="margin-bottom:16px;color:#4cc9f0">Trend</h3><canvas id="c1"></canvas></div>
    <div class="chart-card"><h3 style="margin-bottom:16px;color:#4cc9f0">Distribution</h3><canvas id="c2"></canvas></div>
  </div>
</div>
<script>
var c1=new Chart(document.getElementById('c1'),{{type:'line',data:{{labels:['Jan','Feb','Mar','Apr','May','Jun'],datasets:[{{label:'Value',data:[65,72,80,74,90,88],borderColor:'#4361ee',tension:.4,fill:true,backgroundColor:'rgba(67,97,238,.1)'}}]}},options:{{responsive:true,plugins:{{legend:{{display:false}}}},scales:{{x:{{grid:{{color:'#1e1e35'}}}},y:{{grid:{{color:'#1e1e35'}}}}}}}}}});
var c2=new Chart(document.getElementById('c2'),{{type:'doughnut',data:{{labels:['A','B','C','D'],datasets:[{{data:[30,25,20,25],backgroundColor:['#4361ee','#f72585','#4cc9f0','#06d6a0']}}]}},options:{{responsive:true}}}});
</script>
</body>
</html>""",
}

TEMPLATE_KPI = '<div class="kpi"><div class="kpi-val">{val}</div><div class="kpi-lbl">{label}</div></div>'


def generate_app(description: str, template_type: str = "webapp", lang: str = "en") -> dict:
    """Generate a complete web app from description. Returns dict with html, zip_b64."""
    from intelligence.data_engine import _call_llm

    # Use LLM to customize the template
    prompt = (
        "You are building a web application. Description: '{}'\n"
        "Generate the following as JSON (valid JSON only, no markdown):\n"
        "{{\n"
        "  \"title\": \"App name (2-4 words)\",\n"
        "  \"description\": \"One sentence description\",\n"
        "  \"features\": [\"feature 1\", \"feature 2\", \"feature 3\"],\n"
        "  \"kpis\": [{\"label\": \"Metric\", \"value\": \"0\"}, ...] (4 KPIs if dashboard),\n"
        "  \"content_html\": \"main HTML content for the app body (forms, lists, etc.)\",\n"
        "  \"js\": \"JavaScript functions for interactivity\"\n"
        "}}\n"
        "Make it functional and relevant to: {}"
    ).format(description, description)

    raw = _call_llm(prompt, 1500)
    import json, re
    data = {}
    try:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m: data = json.loads(m.group(0))
    except Exception:
        pass

    title    = data.get("title", "My App")
    desc     = data.get("description", description[:100])
    content  = data.get("content_html", "<p>Welcome to your new app!</p>")
    js       = data.get("js", "")
    kpis_data = data.get("kpis", [{"label":"Users","value":"1,234"},{"label":"Revenue","value":"€45K"},
                                   {"label":"Growth","value":"+12%"},{"label":"Active","value":"892"}])

    kpis_html = "".join(TEMPLATE_KPI.format(val=k.get("value","0"), label=k.get("label","Metric"))
                        for k in kpis_data[:4])

    tmpl = TEMPLATES.get(template_type, TEMPLATES["webapp"])
    html_out = tmpl.format(
        title=title, description=desc,
        content=content, js=js, kpis=kpis_html,
    )

    # Create ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("index.html", html_out)
        zf.writestr("README.md",
            "# {}\n\n{}\n\nGenerated by Zynetrax.\n\n## Setup\nOpen index.html in any browser.\n".format(title, desc))
    zip_b64 = base64.b64encode(zip_buf.getvalue()).decode()

    return {"html": html_out, "zip_b64": zip_b64, "title": title, "description": desc}
