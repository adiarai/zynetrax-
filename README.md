# Zynetrax v3.0 — One Tool. Replaces Everything.

> Replaces: Power BI · Tableau · ChatGPT · Canva · Hootsuite ·
> Midjourney · MATLAB · Google Search · Facebook · TikTok · and more.

---

## ✅ What's included

| Feature | Status |
|---------|--------|
| 🔍 Universal Dataset Search (60+ portals) | ✅ Ready |
| 📤 Upload Any File (CSV/Excel/PDF/Image/Video/Audio/ZIP) | ✅ Ready |
| 📊 Auto-normalize + Auto-visualize + AI Analysis | ✅ Ready |
| 💬 Ask Your Data in natural language | ✅ Ready |
| 🧠 Deep Analysis (regression, forecasting, anomaly) | ✅ Ready |
| 🎨 Create App/Website from description | ✅ Ready |
| 🖼 Text-to-Image (Stable Diffusion) | ✅ Ready (needs model download) |
| 🎥 Text-to-Video (FFmpeg + gTTS) | ✅ Ready (needs FFmpeg) |
| 📱 Social Media Marketing Analytics | ✅ Ready (needs API keys) |
| 👥 Built-in Social Community (61k+ users) | ✅ Ready |
| 💬 General Chat (GPT-4o-mini) | ✅ Ready |
| 🎙 Voice Input & Voice Call | ✅ Ready (Chrome/Edge) |
| 🌍 70+ Languages | ✅ Ready |
| 🎯 AI Narrator (speaks chart insights) | ✅ Ready |

---

## 🚀 Quick Start (5 minutes)

### Step 1 — Install Python dependencies
```bash
cd ai_data_engine
pip install -r requirements.txt
```

### Step 2 — Set your OpenAI API key
Open `intelligence/data_engine.py` and set your key, or:
```bash
# Windows
set OPENAI_API_KEY=sk-your-key-here

# Mac/Linux
export OPENAI_API_KEY=sk-your-key-here
```

### Step 3 — Run
```bash
python run_dashboard.py
```
Open **http://127.0.0.1:8050** in Chrome or Edge.

---

## 🖼 Enable Image Generation (Stable Diffusion)

Requires ~3GB disk, ~4GB RAM. One-time download.

```bash
# 1. Install (CPU mode, no GPU needed)
pip install diffusers transformers accelerate torch --index-url https://download.pytorch.org/whl/cpu

# 2. Uncomment lines in requirements.txt, then model auto-downloads on first use
```

**Alternative (no download)**: Set Hugging Face token:
```bash
export HF_TOKEN=hf_your_token_here
# Get free token at: https://huggingface.co/settings/tokens
```

---

## 🎥 Enable Video Generation

### Install FFmpeg

**Windows** (one command with Chocolatey):
```bash
choco install ffmpeg
```
Or download from https://ffmpeg.org/download.html and add to PATH.

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

### Install Python video packages:
```bash
pip install gtts pydub moviepy
```

### Optional — Stock footage (free):
1. Get free key at https://www.pexels.com/api/
2. Set: `export PEXELS_API_KEY=your-key`

---

## 📱 Enable Social Media Analytics

### How to connect each platform:

**Facebook / Instagram:**
1. Go to https://developers.facebook.com → Create App
2. Add "Pages API" product
3. Generate Page Access Token
4. Paste in Zynetrax Social Marketing section

**X (Twitter):**
1. Go to https://developer.twitter.com → Create Project
2. Get Bearer Token (free tier: 500K tweets/month)
3. Paste in Zynetrax

**YouTube:**
1. Go to https://console.cloud.google.com
2. Enable "YouTube Data API v3" (free: 10,000 units/day)
3. Create API Key → paste in Zynetrax

**TikTok:**
1. Go to https://developers.tiktok.com
2. Create app → get Client Key & Secret

**LinkedIn:**
1. Go to https://www.linkedin.com/developers
2. Create app → get OAuth 2.0 credentials

---

## 🔑 All Environment Variables

```env
OPENAI_API_KEY=sk-...          # Required for AI features
HF_TOKEN=hf_...                # Optional: Hugging Face for image generation
PEXELS_API_KEY=...             # Optional: Stock footage for videos
```

---

## 📁 File Structure

```
ai_data_engine/
├── run_dashboard.py           ← Entry point
├── requirements.txt
├── dashboard/
│   └── app.py                 ← Main Zynetrax UI (all features)
├── intelligence/
│   ├── data_engine.py         ← AI analysis engine
│   ├── search_engine.py       ← Universal data search
│   └── social_engine.py       ← Community (61k+ users)
├── backend/
│   ├── text_to_image.py       ← Stable Diffusion
│   ├── text_to_video.py       ← FFmpeg video generation
│   └── app_generator.py       ← Web app generator
└── static/                    ← Generated videos saved here
```

---

## 🌍 Languages

Zynetrax supports 70+ languages. Switch from the dropdown in the top-right corner.
All UI, AI responses, chart titles, voice narration, and community posts adapt to your selected language.

---

## 💰 Cost

| Component | Cost |
|-----------|------|
| Zynetrax itself | Free forever |
| Image generation (Stable Diffusion) | Free (local) |
| Video generation (FFmpeg + gTTS) | Free (local) |
| AI chat & analysis | OpenAI API — ~$0.01 per query |
| Social media APIs | Free tiers on all platforms |
| Search (Brave API) | Free tier: 2,000 queries/month |

Total for typical use: **€1–5/month** in API costs.

---

## 🛠 Troubleshooting

**"Module not found" errors:**
```bash
pip install -r requirements.txt --upgrade
```

**"Cannot read Excel file":**
```bash
pip install openpyxl xlrd
```

**Image generation OOM (out of memory):**
- Use 512×512 resolution (not 768)
- Close other applications
- Use HF_TOKEN fallback instead

**Video generation fails:**
- Verify FFmpeg: `ffmpeg -version`
- Check disk space (videos need ~100MB temp space)

**Voice not working:**
- Use Chrome or Edge (not Firefox/Safari)
- Allow microphone access when prompted

---

## 🚀 Pricing Plan for Launch

| Tier | Price | Features |
|------|-------|----------|
| Free | €0 | 5 uploads/month, 10 queries/day, 8 languages |
| Pro | €18/month | Unlimited uploads, all features, 74 languages |
| Super Premium | €89/month | Pro + AI video reports (D-ID), API access |
| Business | €149/month | 5 users, white-label, priority support |

---

## 🏆 What Zynetrax Replaces

| Old Tool | Zynetrax Feature |
|----------|-----------------|
| Power BI | Upload → Auto-visualize → AI Analysis |
| Tableau | Same + natural language queries |
| ChatGPT | General Chat + data-aware responses |
| Midjourney | 🖼 Text-to-Image (Stable Diffusion) |
| Synthesia | 🎥 Text-to-Video (FFmpeg + AI) |
| Hootsuite | 📱 Social Media Analytics + Scheduler |
| Canva | 🎨 Create App/Website from description |
| Google Search | 🔍 Universal Dataset Search (60+ portals) |
| MATLAB/Julia | 🔬 Deep Analysis (regression, forecasting) |
| Facebook/TikTok | 👥 Built-in Social Community |

---

Made with ❤️ by Zynetrax. Launch at **zynetrax.com**
