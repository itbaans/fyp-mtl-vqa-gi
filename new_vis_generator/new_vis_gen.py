"""
GI Endoscopy ROI Caption Generator
------------------------------------
Accepts an annotated image (contour + bbox drawn) and an ROI type,
builds the appropriate prompt, and returns a short visual caption
using Gemma 3 27B via the Google Gemini API.

Supported ROI types:
    - z-line          (landmark)
    - cecum           (landmark)
    - oesophagitis    (abnormality)
    - ulcerative-colitis (abnormality)
    - polyp           (polyp)
    - instrument      (instrument)

Usage:
    python gi_caption_generator.py --image path/to/image.jpg --roi-type z-line
    python gi_caption_generator.py --image path/to/image.jpg --roi-type polyp

Install deps:
    pip install google-genai pillow
"""

import argparse
import base64
import os
import sys
from pathlib import Path

# pip install google-genai
from google import genai
from google.genai import types


# ─────────────────────────────────────────────
# ROI taxonomy
# ─────────────────────────────────────────────

ROI_TAXONOMY = {
    # subtype -> (display_name, category)
    "z-line":             ("Z-line",             "landmark"),
    "cecum":              ("Cecum",              "landmark"),
    "oesophagitis":       ("Oesophagitis",       "abnormality"),
    "ulcerative-colitis": ("Ulcerative colitis", "abnormality"),
    "polyp":              ("Polyp",              "polyp"),
    "instrument":         ("Instrument",         "instrument"),
}


# ─────────────────────────────────────────────
# Per-category visual focus hints
# ─────────────────────────────────────────────

CATEGORY_HINTS = {
    "landmark": (
        "Focus on the visible boundary or transition zone. "
        "Describe any color change, difference in surface texture, "
        "or structural feature (such as a fold or ring-like edge) "
        "that makes this location visually distinct."
    ),

    "abnormality": (
        "Focus on visible surface changes inside the marked region. "
        "Note any redness, raw or eroded patches, unusual color compared "
        "to surrounding tissue, signs of bleeding or dark discoloration, "
        "swelling, or irregular texture such as granularity or roughness."
    ),

    "polyp": (
        "Focus on the shape of the marked structure — whether it is raised, "
        "flat, or has a stalk — and describe its surface color and texture. "
        "Note how it stands out from the surrounding lining."
    ),

    "instrument": (
        "Focus on the physical appearance of the object inside the marked region. "
        "Describe its shape, color, and surface material, and how it contrasts "
        "with the surrounding tissue."
    ),
}


# ─────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a visual description assistant for gastrointestinal endoscopy images.

You will be given an endoscopy image with a bounding box and contour drawn around a region of interest (ROI), along with the type and name of that region.

Write a short visual description of what is visible inside the marked region. Follow these rules strictly:
- Maximum 2 sentences.
- Use plain, everyday language — no medical or clinical terms.
- Describe only what you can see: color, texture, shape, surface detail, boundaries.
- Do not diagnose, speculate about cause, or repeat the ROI label.
- Do not describe the bounding box or contour lines themselves.
- CRITICAL: Start your response directly with the visual description. Do NOT use introductory filler phrases like "The marked area is...", "The area looks like...", "Inside the marked area...", or "This area shows...". Just describe the features immediately (e.g., "A slightly raised, bumpy patch with a reddish color...")."""


def build_user_prompt(display_name: str, category: str) -> str:
    hint = CATEGORY_HINTS[category]
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"ROI type: {category.capitalize()}\n"
        f"ROI name: {display_name}\n\n"
        f"The bounding box and contour in the image mark the region of interest. "
        f"Provide a direct visual description of the highlighted area in 1–2 sentences without any introductory filler.\n\n"
        f"{hint}"
    )


# ─────────────────────────────────────────────
# Image loader
# ─────────────────────────────────────────────

def load_image_as_base64(image_path: str) -> tuple[str, str]:
    """Returns (base64_data, mime_type)."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime_map = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".bmp":  "image/bmp",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(suffix, "image/jpeg")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, mime_type


# ─────────────────────────────────────────────
# API call
# ─────────────────────────────────────────────

def generate_caption(
    image_path: str,
    roi_type: str,
    api_key: str | None = None,
) -> str:
    """
    Main entry point. Returns the generated caption string.

    Args:
        image_path: Path to the annotated image (with contour + bbox drawn).
        roi_type:   One of the supported ROI type strings (see ROI_TAXONOMY).
        api_key:    Google Gemini API key. Falls back to GEMINI_API_KEY env var.
    """

    # ── Resolve ROI
    roi_type_key = roi_type.lower().strip()
    if roi_type_key not in ROI_TAXONOMY:
        supported = ", ".join(ROI_TAXONOMY.keys())
        raise ValueError(
            f"Unknown ROI type '{roi_type}'. Supported: {supported}"
        )
    display_name, category = ROI_TAXONOMY[roi_type_key]

    # ── Resolve API key
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "No API key found. Set GEMINI_API_KEY env var or pass --api-key."
        )

    # ── Build prompts
    user_prompt = build_user_prompt(display_name, category)

    # ── Load image
    image_b64, mime_type = load_image_as_base64(image_path)

    # ── Call Gemma 3 27B via Gemini API
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            mime_type=mime_type,
                            data=image_b64,
                        )
                    ),
                    types.Part(text=user_prompt),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=120,   # hard cap — captions should be short
            temperature=0.2,         # low temp = more factual, less hallucination
        ),
    )

    caption = response.text.strip()
    return caption


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a short visual caption for a GI endoscopy ROI."
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to the annotated image (contour + bbox drawn).",
    )
    parser.add_argument(
        "--roi-type", "-r",
        required=True,
        choices=list(ROI_TAXONOMY.keys()),
        metavar="ROI_TYPE",
        help=f"ROI type. One of: {', '.join(ROI_TAXONOMY.keys())}",
    )
    parser.add_argument(
        "--api-key", "-k",
        default=None,
        help="Google Gemini API key. Defaults to GEMINI_API_KEY env var.",
    )
    args = parser.parse_args()

    try:
        caption = generate_caption(
            image_path=args.image,
            roi_type=args.roi_type,
            api_key=args.api_key,
        )
        print(f"\nROI     : {args.roi_type}")
        print(f"Caption : {caption}\n")

    except (ValueError, EnvironmentError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()