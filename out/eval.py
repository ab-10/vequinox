import argparse
import os
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from shared import PREFIX

MODEL_NAME = "Qwen/Qwen3-0.6B"
SYS_PROMPT = "You are an SVG generator. Respond only with valid SVG code. /no_think"
DEFAULT_PROMPT = "Generate SVG code of a pelican riding a bicycle"
DEFAULT_CHECKPOINT_DIR = "./vequinox-checkpoints"


def get_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the latest checkpoint in the checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    checkpoints = [d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not checkpoints:
        return None

    latest = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
    return str(latest)


def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer from a path."""
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.chat_template = tokenizer.chat_template.replace(
        "<|im_start|>assistant", "<|im_start|>assistant " + PREFIX
    )
    return model, tokenizer


def generate_svg(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate SVG from a prompt using the model."""
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": prompt},
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def extract_svg_content(generated_text: str) -> str:
    """Extract the SVG content from generated text."""
    import re

    match = re.search(r"(<svg.*?</svg>)", generated_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)

    if "<svg" in generated_text.lower():
        start_idx = generated_text.lower().find("<svg")
        return generated_text[start_idx:]

    return generated_text


def save_svg(content: str, filepath: str):
    """Save SVG content to a file."""
    with open(filepath, "w") as f:
        f.write(content)
    print(f"Saved SVG to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SVG generation from original and checkpoint models")
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"Prompt for SVG generation (default: {DEFAULT_PROMPT})",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Directory containing checkpoints (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint to use (default: latest)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_outputs",
        help="Directory to save output SVGs (default: ./eval_outputs)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate (default: 512)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 50)
    print("Loading original model...")
    print("=" * 50)
    original_model, original_tokenizer = load_model_and_tokenizer(MODEL_NAME)

    print("\nGenerating SVG from original model...")
    original_output = generate_svg(original_model, original_tokenizer, args.prompt, args.max_new_tokens)
    original_svg = extract_svg_content(original_output)

    original_svg_path = os.path.join(args.output_dir, "original.svg")
    save_svg(original_svg, original_svg_path)

    del original_model
    del original_tokenizer

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = get_latest_checkpoint(args.checkpoint_dir)

    if checkpoint_path is None:
        print(f"\nNo checkpoints found in {args.checkpoint_dir}")
        print("Only original model SVG was generated.")
        return

    print("\n" + "=" * 50)
    print(f"Loading checkpoint model from {checkpoint_path}...")
    print("=" * 50)
    checkpoint_model, checkpoint_tokenizer = load_model_and_tokenizer(checkpoint_path)

    print("\nGenerating SVG from checkpoint model...")
    checkpoint_output = generate_svg(checkpoint_model, checkpoint_tokenizer, args.prompt, args.max_new_tokens)
    checkpoint_svg = extract_svg_content(checkpoint_output)

    checkpoint_svg_path = os.path.join(args.output_dir, "checkpoint.svg")
    save_svg(checkpoint_svg, checkpoint_svg_path)

    print("\n" + "=" * 50)
    print("Evaluation complete!")
    print("=" * 50)
    print(f"Original model SVG: {original_svg_path}")
    print(f"Checkpoint model SVG: {checkpoint_svg_path}")


if __name__ == "__main__":
    main()
