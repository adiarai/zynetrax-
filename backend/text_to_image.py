"""
backend/text_to_image.py — Stable Diffusion image generation
Free, local, no API costs.
"""
import os
import io
import base64

_pipeline = None

def _load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "stabilityai/stable-diffusion-2-1-base"
        _pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        _pipeline.safety_checker = None
        print("[ImageGen] Pipeline loaded on", device)
        return _pipeline
    except Exception as e:
        print("[ImageGen] Load error:", e)
        raise

def generate_image(prompt: str, negative_prompt: str = "", resolution: int = 512) -> str:
    """
    Generate image from prompt.
    Returns base64-encoded PNG string.
    Falls back to HF Inference API if local model fails.
    """
    resolution = min(max(resolution, 256), 768)

    # Try local SD pipeline first
    try:
        pipe = _load_pipeline()
        result = pipe(
            prompt,
            negative_prompt=negative_prompt or "blurry, bad quality, watermark, text",
            height=resolution,
            width=resolution,
            num_inference_steps=20,
            guidance_scale=7.5,
        )
        img = result.images[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as local_err:
        print("[ImageGen] Local failed:", local_err)

    # Fallback: HF Inference API
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        try:
            import requests
            headers = {"Authorization": "Bearer {}".format(hf_token)}
            payload = {"inputs": prompt, "parameters": {"negative_prompt": negative_prompt}}
            resp = requests.post(
                "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1",
                headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                return base64.b64encode(resp.content).decode()
        except Exception as api_err:
            print("[ImageGen] HF API failed:", api_err)

    raise RuntimeError(
        "Image generation failed. To enable: install diffusers+torch, or set HF_TOKEN env var.\n"
        "Install: pip install diffusers torch transformers accelerate"
    )
