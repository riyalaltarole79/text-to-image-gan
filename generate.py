"""
generate.py
===========
CLI inference: generate images from text prompts.

Examples
--------
    # Single prompt
    python generate.py --text "a vibrant red tulip in sunlight"

    # Multiple prompts from file
    python generate.py --prompts_file prompts.txt --n_per_text 4

    # Interpolation between two prompts
    python generate.py \
        --interpolate \
        --text_a "red fire lily" \
        --text_b "blue iris flower" \
        --steps 10

    # Custom output path
    python generate.py --text "purple lavender" --output outputs/lavender.png
"""

import argparse
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(__file__))

from src.config import CFG
from src.inference import TextToImagePipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CHECKPOINT = "checkpoints/latest.pt"
EMBEDDER   = "checkpoints/embedder.pkl"


def parse_args():
    parser = argparse.ArgumentParser(description="Text-to-Image generation")
    # Input
    parser.add_argument("--text",          type=str, default=None,
                        help="Single text prompt")
    parser.add_argument("--prompts_file",  type=str, default=None,
                        help="Path to .txt file with one prompt per line")
    # Interpolation
    parser.add_argument("--interpolate",   action="store_true")
    parser.add_argument("--text_a",        type=str, default=None)
    parser.add_argument("--text_b",        type=str, default=None)
    parser.add_argument("--steps",         type=int, default=8)
    # Generation params
    parser.add_argument("--n_per_text",    type=int, default=1)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--output",        type=str, default=None)
    # Model
    parser.add_argument("--checkpoint",    type=str, default=CHECKPOINT)
    parser.add_argument("--embedder",      type=str, default=EMBEDDER)
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validation ────────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        print("        Run  python train.py  first.")
        sys.exit(1)

    # ── Load pipeline ─────────────────────────────────────────────────────
    pipe = TextToImagePipeline.from_checkpoint(
        args.checkpoint,
        args.embedder,
        cfg=CFG,
    )

    # ── Interpolation mode ────────────────────────────────────────────────
    if args.interpolate:
        if not args.text_a or not args.text_b:
            print("[ERROR] --interpolate requires --text_a and --text_b")
            sys.exit(1)
        print(f"Interpolating: '{args.text_a}'  →  '{args.text_b}'  ({args.steps} steps)")
        imgs = pipe.interpolate(args.text_a, args.text_b, steps=args.steps)
        out  = args.output or f"outputs/interp_{args.steps}steps.png"
        pipe.save(imgs, out, nrow=args.steps)
        print(f"Saved → {out}")
        return

    # ── Collect prompts ───────────────────────────────────────────────────
    prompts = []
    if args.text:
        prompts = [args.text]
    elif args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Default demo prompts
        prompts = [
            "a vibrant red rose with soft petals in sunlight",
            "purple lavender in a meadow with green leaves",
            "a delicate white daisy with a golden center",
            "a beautiful pink primrose in full bloom",
        ]
        print("No prompt given. Using demo prompts.")

    print(f"Generating {len(prompts) * args.n_per_text} image(s) …")
    imgs = pipe.generate(prompts, n_per_text=args.n_per_text, seed=args.seed)

    # ── Build caption labels for display ─────────────────────────────────
    labels = [p for p in prompts for _ in range(args.n_per_text)]

    # ── Save ─────────────────────────────────────────────────────────────
    out = args.output or "outputs/generated.png"
    pipe.save(imgs, out, captions=labels)
    print(f"Done → {out}")


if __name__ == "__main__":
    main()
