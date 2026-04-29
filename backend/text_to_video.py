"""
backend/text_to_video.py — FFmpeg-based video generation
Script → Voice (gTTS) → Visuals → MP4
Free, local, no API costs.
"""
import os
import re
import json
import base64
import tempfile
import subprocess
import uuid


def _call_llm(prompt: str, max_tokens: int = 800) -> str:
    try:
        from intelligence.data_engine import _call_llm as _llm
        return _llm(prompt, max_tokens)
    except Exception:
        return ""


def _scene_from_llm(script: str, style: str, lang: str) -> list:
    """Break script into scenes using LLM."""
    lang_names = {"en":"English","de":"German","fr":"French","es":"Spanish",
                  "ar":"Arabic","zh":"Chinese","hi":"Hindi","pt":"Portuguese"}
    ln = lang_names.get(lang, "English")
    prompt = (
        "Break this video script into 3-6 scenes. Respond ONLY in valid JSON.\n"
        "Script: '{}'\nStyle: {}\nLanguage: {}\n\n"
        "Return a JSON array where each object has:\n"
        "  text: spoken narration for this scene (in {})\n"
        "  keyword: 2-3 word visual search term in English\n"
        "  duration: seconds (2-8 based on text length)\n"
        "Return ONLY the JSON array, no markdown."
    ).format(script[:800], style, ln, ln)
    raw = _call_llm(prompt, 600)
    if not raw:
        return _fallback_scenes(script)
    try:
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return _fallback_scenes(script)


def _fallback_scenes(script: str) -> list:
    """Simple sentence-based scene split."""
    sentences = [s.strip() for s in re.split(r'[.!?]', script) if len(s.strip()) > 15]
    scenes = []
    for i, sent in enumerate(sentences[:5]):
        words = sent.split()
        kw = " ".join(words[:3]) if len(words) >= 3 else sent
        scenes.append({"text": sent, "keyword": kw, "duration": max(3, len(sent)//10)})
    return scenes or [{"text": script[:200], "keyword": "nature landscape", "duration": 5}]


def _tts(text: str, lang: str, out_path: str) -> bool:
    """Generate speech audio. Returns True on success."""
    # Try gTTS
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang if len(lang) == 2 else "en", slow=False)
        tts.save(out_path)
        return True
    except Exception:
        pass
    # Fallback: pyttsx3 (offline)
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        return os.path.exists(out_path)
    except Exception:
        pass
    return False


def _fetch_stock_image(keyword: str, out_path: str) -> bool:
    """Fetch free stock image from Pexels or Pixabay."""
    pexels_key = os.environ.get("PEXELS_API_KEY", "")
    if pexels_key:
        try:
            import requests
            resp = requests.get(
                "https://api.pexels.com/v1/search",
                headers={"Authorization": pexels_key},
                params={"query": keyword, "per_page": 1, "size": "medium"},
                timeout=10)
            if resp.status_code == 200:
                photos = resp.json().get("photos", [])
                if photos:
                    img_url = photos[0]["src"]["medium"]
                    img_resp = requests.get(img_url, timeout=15)
                    if img_resp.status_code == 200:
                        with open(out_path, "wb") as f:
                            f.write(img_resp.content)
                        return True
        except Exception:
            pass
    return False


def _make_title_image(text: str, out_path: str, width: int = 1280, height: int = 720) -> bool:
    """Create a title card image using Pillow."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (width, height), color=(10, 10, 20))
        draw = ImageDraw.Draw(img)
        # Gradient-like overlay
        for i in range(height):
            alpha = int(40 * (i / height))
            draw.line([(0, i), (width, i)], fill=(67, 97, 238, alpha))
        # Text
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
        except Exception:
            font = ImageFont.load_default()
            small_font = font
        # Word wrap
        words = text.split()
        lines, line = [], []
        for w in words:
            test = " ".join(line + [w])
            if len(test) > 40 and line:
                lines.append(" ".join(line)); line = [w]
            else:
                line.append(w)
        if line: lines.append(" ".join(line))
        y = height//2 - len(lines)*30
        for ln in lines:
            try:
                bbox = draw.textbbox((0,0), ln, font=font)
                tw = bbox[2] - bbox[0]
            except Exception:
                tw = len(ln) * 24
            x = (width - tw) // 2
            # Shadow
            draw.text((x+2, y+2), ln, fill=(0,0,0), font=font)
            draw.text((x, y), ln, fill=(232,232,240), font=font)
            y += 60
        img.save(out_path, "PNG")
        return True
    except Exception as e:
        print("[TitleImage]", e)
        return False


def _ffmpeg_available() -> bool:
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def generate_video(script: str, style: str = "educational",
                   music: str = "none", lang: str = "en") -> str:
    """
    Generate MP4 video from script.
    Returns path to output video file.
    """
    if not _ffmpeg_available():
        raise RuntimeError(
            "FFmpeg not found. Install from https://ffmpeg.org/download.html\n"
            "Windows: choco install ffmpeg  |  Mac: brew install ffmpeg  |  Linux: apt install ffmpeg"
        )

    tmpdir = tempfile.mkdtemp(prefix="zynetrax_video_")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "video_{}.mp4".format(uuid.uuid4().hex[:8]))

    print("[VideoGen] Parsing scenes…")
    scenes = _scene_from_llm(script, style, lang)
    print("[VideoGen] {} scenes".format(len(scenes)))

    scene_videos = []

    for i, scene in enumerate(scenes):
        print("[VideoGen] Scene {}/{}: {}…".format(i+1, len(scenes), scene.get("text","")[:40]))
        scene_dir = os.path.join(tmpdir, "scene_{}".format(i))
        os.makedirs(scene_dir, exist_ok=True)

        # 1. Generate voice
        audio_path = os.path.join(scene_dir, "audio.mp3")
        has_audio  = _tts(scene.get("text",""), lang[:2], audio_path)

        # 2. Get visual
        img_path = os.path.join(scene_dir, "frame.png")
        has_img  = False
        if scene.get("keyword"):
            has_img = _fetch_stock_image(scene["keyword"], img_path)
        if not has_img:
            has_img = _make_title_image(scene.get("text","")[:100], img_path)

        # 3. Compose scene video
        duration = scene.get("duration", 4)
        scene_out = os.path.join(scene_dir, "scene.mp4")

        if has_img and has_audio:
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", img_path,
                "-i", audio_path,
                "-c:v", "libx264", "-tune", "stillimage",
                "-c:a", "aac", "-b:a", "128k",
                "-pix_fmt", "yuv420p",
                "-shortest", "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
                scene_out,
            ]
        elif has_img:
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", img_path,
                "-c:v", "libx264", "-t", str(duration),
                "-pix_fmt", "yuv420p",
                "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
                scene_out,
            ]
        else:
            # Black screen with text
            _make_title_image(scene.get("text","")[:80], img_path)
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", img_path,
                "-c:v", "libx264", "-t", str(duration),
                "-pix_fmt", "yuv420p",
                scene_out,
            ]

        try:
            subprocess.run(cmd, capture_output=True, timeout=60, check=True)
            scene_videos.append(scene_out)
        except Exception as e:
            print("[VideoGen] Scene {} failed: {}".format(i, e))

    if not scene_videos:
        raise RuntimeError("No scenes generated. Check FFmpeg installation.")

    # 4. Concatenate all scenes
    list_file = os.path.join(tmpdir, "concat.txt")
    with open(list_file, "w") as f:
        for v in scene_videos:
            f.write("file '{}'\n".format(v.replace("\\", "/")))

    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            output_path,
        ], capture_output=True, timeout=120, check=True)
        print("[VideoGen] Done:", output_path)
        return "/static/{}".format(os.path.basename(output_path))
    except Exception as e:
        raise RuntimeError("Video concatenation failed: {}".format(e))
