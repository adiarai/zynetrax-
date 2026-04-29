"""
social_engine.py  — Zynetrax Community Engine
=============================================
Place at: ai_data_engine/intelligence/social_engine.py

Provides:
  - 61k+ simulated community members
  - Live post feed (dummy + real)
  - Auto-reply to real user posts via GPT-4o-mini
  - Trending topics & leaderboard
"""

import random
import hashlib
import time
from datetime import datetime, timedelta

# ── Constants ────────────────────────────────────────────────────────
FIRST_NAMES = [
    "James","Sophia","Liam","Emma","Noah","Olivia","Ethan","Ava","Lucas","Mia",
    "Mason","Isabella","Logan","Charlotte","Oliver","Amelia","Aiden","Harper",
    "Muhammad","Fatima","Ahmed","Aisha","Ali","Zara","Raj","Priya","Arjun","Neha",
    "Wei","Mei","Jing","Fang","Carlos","Maria","Juan","Ana","Pedro","Sofia",
    "Yuki","Kenji","Sakura","Haruto","Omar","Layla","Felix","Anna","Leon","Lena",
    "Pierre","Camille","Antoine","Lucie","Hugo","Giorgio","Giulia","Marco","Chiara",
    "Daniel","Hannah","Alexander","Benjamin","Elias","Adam","Nathan","Ryan","Tyler",
    "Jayden","Ella","Grace","Chloe","Lily","Zoe","Amara","Kofi","Kwame","Nia",
]
LAST_NAMES = [
    "Smith","Johnson","Williams","Brown","Jones","Miller","Davis","Wilson","Moore",
    "Anderson","Thomas","Jackson","White","Harris","Martin","Thompson","Garcia",
    "Khan","Ahmed","Ali","Hassan","Patel","Shah","Sharma","Singh","Kumar",
    "Wang","Li","Zhang","Chen","Liu","Yang","Huang","Zhao",
    "Müller","Schmidt","Fischer","Weber","Meyer","Wagner","Becker","Schulz",
    "Rossi","Ferrari","Russo","Bianchi","Esposito","Romano","Colombo",
    "García","Martínez","González","López","Rodríguez","Pérez","Sánchez",
    "Santos","Oliveira","Silva","Pereira","Costa","Ferreira","Ribeiro",
    "Okonkwo","Mensah","Diallo","Nkosi","Abebe","Kimani","Osei","Asante",
]
DOMAINS = ["gmail.com","yahoo.com","outlook.com","hotmail.com","protonmail.com","icloud.com","fastmail.com"]
COUNTRIES = [
    "USA","Germany","UK","France","India","Pakistan","UAE","Saudi Arabia",
    "Brazil","Canada","Australia","Japan","South Korea","China","Italy","Spain",
    "Netherlands","Sweden","Poland","Turkey","Egypt","Nigeria","South Africa",
    "Mexico","Argentina","Indonesia","Malaysia","Philippines","Kenya","Ghana",
]
ROLES = [
    "Data Analyst","Data Scientist","BI Developer","Research Analyst",
    "Financial Analyst","Product Manager","Software Engineer","ML Engineer",
    "Marketing Analyst","Operations Manager","Consultant","Student","Professor",
    "Economist","Statistician","Epidemiologist","Supply Chain Analyst",
    "Quantitative Researcher","Healthcare Analyst","Risk Manager","PhD Researcher",
]
AVATAR_COLORS = [
    "#4361ee","#f72585","#4cc9f0","#7209b7","#06d6a0",
    "#ffd166","#ef476f","#118ab2","#26a269","#e76f51",
    "#a8dadc","#457b9d","#e63946","#2ec4b6","#ff9f1c",
]
TOPICS = [
    "sales","healthcare","climate","financial","education","crime","economic",
    "population","energy","supply chain","marketing","HR","real estate",
    "stock market","social media","genomics","traffic","agriculture","COVID-19",
    "customer churn","sports","election","sentiment","logistics","retail",
]
COLS = [
    "revenue","sales volume","temperature","GDP","population","mortality rate",
    "conversion rate","churn rate","satisfaction score","price","quantity","profit margin",
]
METHODS = [
    "regression","clustering","time series","PCA","anomaly detection","forecasting",
    "correlation","classification","segmentation","trend analysis",
]
OLD_TOOLS = ["Excel","Power BI","Tableau","Python scripts","R","SPSS","manual reports","Google Sheets"]

POST_TEMPLATES = [
    "Just finished analysing {topic} data — found some really surprising patterns in {col1}! Anyone else working on this? 📊",
    "Used Zynetrax to visualise {topic} and the AI narrator feature blew my mind 🎯",
    "Question for the community: best approach for handling missing values in {topic} datasets?",
    "Sharing my {topic} analysis — the correlation between {col1} and {col2} was much stronger than expected",
    "Running a {method} on {topic} right now. Zynetrax makes it so easy compared to {old_tool} 🙌",
    "Pro tip: when uploading large {topic} CSVs, the auto-normalisation handles date columns perfectly",
    "My team switched from {old_tool} to Zynetrax for all our {topic} dashboards. Saving 6+ hours/week ⏱️",
    "The deep analysis feature just found an anomaly in our {topic} data that we completely missed 🧠",
    "Anyone have good sources for {topic} open datasets? The search feature is great but I need more historical data",
    "Regression on {topic} with R² of 0.87 — not bad! The prediction chart was very clean ✅",
    "Uploaded a messy {topic} Excel file and Zynetrax auto-fixed all the column types. Huge time saver 💪",
    "The AI explanation of the {topic} scatter plot was more insightful than my manual analysis tbh 😅",
    "Teaching my students about {topic} using Zynetrax — they love the voice narrator feature 🎓",
    "Quick benchmark: Zynetrax built a {topic} dashboard in 12 sec. {old_tool} took me 45 min manually ⚡",
    "The multilingual support is a game changer for our international {topic} reporting 🌍",
    "Finally visualised our {topic} data properly — the auto-explore gave us 4 insightful charts instantly",
    "Who else uses Zynetrax for {method}? The results beat what I was getting with {old_tool}",
    "The {topic} dataset I downloaded from the search feature was perfect — saved me hours of searching",
    "Science & Engineering mode solved a {method} problem that was taking me days in {old_tool} 🔬",
    "Impressed by how the AI narrator explained the {col1} trend — it caught a seasonal pattern I missed",
]
COMMENT_TEMPLATES = [
    "Great analysis! I've been working on something similar with {topic} data.",
    "Have you tried the deep analysis feature? It would be perfect for this.",
    "This is exactly what I needed. Thanks for sharing! 🙏",
    "Really interesting finding. Did you check if the trend holds after {year}?",
    "We had the same issue. Auto-normalize fixed it instantly.",
    "The correlation makes sense when you consider {factor} effects.",
    "Bookmarked! I'll share it with my team — we're working on {topic} too.",
    "Nice work! What sample size are you working with?",
    "I'd love to see the full scatter plot for this. Mind sharing the viz?",
    "This is why I switched from {old_tool} to Zynetrax 😄",
    "Interesting approach. Have you considered using {method} instead?",
    "Thanks! The AI narrator explaining this would be 🔥",
    "Following this thread — very relevant to my current project.",
    "That R² score is excellent for this type of data. Well done!",
    "Great point! We published something on this {topic} correlation last year.",
]
FACTORS = ["seasonality","inflation","demographic shift","policy change","market cycle","technology adoption"]


def _uid(seed: str) -> str:
    return hashlib.md5(seed.encode()).hexdigest()[:12]


def _time_ago(dt: datetime) -> str:
    diff = datetime.now() - dt
    s = diff.total_seconds()
    if s < 60:   return "just now"
    if s < 3600: return f"{int(s//60)}m ago"
    if s < 86400: return f"{int(s//3600)}h ago"
    if s < 604800: return f"{int(s//86400)}d ago"
    return dt.strftime("%b %d")


def generate_dummy_user(idx: int) -> dict:
    rng = random.Random(idx * 7919)
    first    = rng.choice(FIRST_NAMES)
    last     = rng.choice(LAST_NAMES)
    num      = rng.randint(1, 999)
    username = f"{first.lower()}{last.lower()}{num}"
    joined_dt = datetime.now() - timedelta(days=rng.randint(1, 730))
    return {
        "id":        _uid(username),
        "username":  username,
        "name":      f"{first} {last}",
        "email":     f"{username}@{rng.choice(DOMAINS)}",
        "country":   rng.choice(COUNTRIES),
        "role":      rng.choice(ROLES),
        "joined":    joined_dt.strftime("%Y-%m-%d"),
        "color":     rng.choice(AVATAR_COLORS),
        "followers": rng.randint(0, 2400),
        "posts":     rng.randint(1, 85),
        "initials":  first[0].upper() + last[0].upper(),
        "is_dummy":  True,
    }


# Pre-generate pool of 500 users (simulates 61k by cycling)
_USER_POOL: list = [generate_dummy_user(i) for i in range(500)]


def get_dummy_user(idx: int) -> dict:
    return _USER_POOL[idx % len(_USER_POOL)]


def get_dummy_user_count() -> int:
    return 61_247


def generate_post(post_idx: int, user: dict) -> dict:
    rng = random.Random(post_idx * 3571)
    tmpl = rng.choice(POST_TEMPLATES)
    col1  = rng.choice(COLS)
    col2  = rng.choice([c for c in COLS if c != col1])
    text = tmpl.format(
        topic=rng.choice(TOPICS), col1=col1, col2=col2,
        method=rng.choice(METHODS), old_tool=rng.choice(OLD_TOOLS),
    )
    hours_ago = rng.randint(0, 96)
    dt = datetime.now() - timedelta(hours=hours_ago, minutes=rng.randint(0, 59))
    return {
        "id":        _uid(f"post_{post_idx}"),
        "user_id":   user["id"],
        "username":  user["username"],
        "name":      user["name"],
        "initials":  user["initials"],
        "color":     user["color"],
        "role":      user["role"],
        "country":   user["country"],
        "text":      text,
        "timestamp": dt.isoformat(),
        "time_ago":  _time_ago(dt),
        "likes":     rng.randint(0, 142),
        "comments":  rng.randint(0, 18),
        "is_dummy":  True,
    }


def get_feed_posts(page: int = 0, per_page: int = 15) -> list:
    posts = []
    for i in range(per_page):
        idx  = page * per_page + i
        user = get_dummy_user(idx * 13 + 7)
        posts.append(generate_post(idx, user))
    return posts


def generate_comment(post_id: str, comment_idx: int, is_reply: bool = False) -> dict:
    rng  = random.Random(abs(hash(post_id + str(comment_idx))) % (2**31))
    user = get_dummy_user(rng.randint(0, 499))
    tmpl = rng.choice(COMMENT_TEMPLATES)
    text = tmpl.format(
        topic=rng.choice(TOPICS),
        year=rng.randint(2019, 2025),
        old_tool=rng.choice(OLD_TOOLS),
        factor=rng.choice(FACTORS),
        method=rng.choice(METHODS),
    )
    minutes_ago = rng.randint(1, 180) if not is_reply else rng.randint(1, 60)
    dt = datetime.now() - timedelta(minutes=minutes_ago)
    return {
        "id":        _uid(f"comment_{post_id}_{comment_idx}"),
        "post_id":   post_id,
        "user_id":   user["id"],
        "username":  user["username"],
        "name":      user["name"],
        "initials":  user["initials"],
        "color":     user["color"],
        "text":      text,
        "time_ago":  _time_ago(dt),
        "likes":     rng.randint(0, 24),
        "is_reply":  is_reply,
        "is_dummy":  True,
    }


def get_post_comments(post_id: str, count: int = 3) -> list:
    return [generate_comment(post_id, i) for i in range(count)]


def generate_auto_reply(real_post_text: str, reply_idx: int = 0) -> dict:
    """Generate a contextual auto-reply to a real user's post using GPT-4o-mini."""
    rng  = random.Random(int(time.time()) + reply_idx * 17)
    user = get_dummy_user(rng.randint(0, 499))

    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=80,
            messages=[
                {"role": "system", "content": (
                    "You are a data analyst community member on Zynetrax. "
                    "Write a SHORT natural reply (1-2 sentences, max 80 words). "
                    "Sound genuinely interested. Use 1 emoji occasionally. "
                    "Be specific to the post. Do NOT say you're an AI."
                )},
                {"role": "user", "content": real_post_text},
            ],
        )
        reply_text = resp.choices[0].message.content.strip()
    except Exception:
        fallbacks = [
            "Really interesting! Had a similar experience with my dataset 📊",
            "Thanks for sharing — super relevant to what I'm working on!",
            "Great insight! Have you tried the AI narrator feature for this?",
            "This is exactly what I've been looking for 🙌",
            "Nice work! What dataset are you using?",
            "Bookmarked — sharing with my team later.",
        ]
        reply_text = rng.choice(fallbacks)

    dt = datetime.now() - timedelta(seconds=rng.randint(5, 90))
    return {
        "id":        _uid(f"autoreply_{reply_idx}_{time.time()}"),
        "user_id":   user["id"],
        "username":  user["username"],
        "name":      user["name"],
        "initials":  user["initials"],
        "color":     user["color"],
        "text":      reply_text,
        "time_ago":  "just now",
        "likes":     rng.randint(0, 8),
        "is_dummy":  True,
    }


def get_trending_topics() -> list:
    base = [
        {"tag": "MachineLearning", "posts": 4821},
        {"tag": "DataScience",     "posts": 3956},
        {"tag": "AI",              "posts": 3412},
        {"tag": "Python",          "posts": 2988},
        {"tag": "Visualization",   "posts": 2341},
        {"tag": "Statistics",      "posts": 1876},
        {"tag": "NLP",             "posts": 1654},
        {"tag": "Finance",         "posts": 1432},
        {"tag": "Healthcare",      "posts": 1198},
        {"tag": "Climate",         "posts": 987},
    ]
    for t in base:
        t["posts"] += random.randint(-20, 30)
    return base


def get_top_contributors(n: int = 8) -> list:
    out = []
    for i in range(n):
        u = get_dummy_user(i * 23 + 5)
        out.append({
            **u,
            "analyses_shared": random.randint(12, 150),
            "helpful_votes":   random.randint(50, 800),
            "streak_days":     random.randint(1, 90),
        })
    return sorted(out, key=lambda x: x["helpful_votes"], reverse=True)
