import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from shared import MODEL_NAME, PREFIX, PROMPT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with trained vequinox model"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True, help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_svgs",
        help="Directory to save generated SVGs",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1, help="Number of SVGs to generate"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate"
    )
    return parser.parse_args()


def extract_svg(text: str) -> str:
    """Extract SVG content from model output."""
    # The model output starts with PREFIX which contains the svg opening tag
    # We need to find the closing </svg> tag
    svg_start = text.find("<svg")
    if svg_start == -1:
        return text

    svg_end = text.find("</svg>", svg_start)
    if svg_end == -1:
        # No closing tag, return from start to end
        return text[svg_start:]

    return text[svg_start : svg_end + len("</svg>")]


def main():
    args = parse_args()

    print(f"Loading model from checkpoint: {args.checkpoint_dir}")

    # Load tokenizer from base model, model from checkpoint
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint_dir)

    # Apply the same chat template modification as training
    tokenizer.chat_template = tokenizer.chat_template.replace(
        "<|im_start|>assistant", "<|im_start|>assistant " + PREFIX
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get checkpoint name for output file naming
    checkpoint_name = Path(args.checkpoint_dir).name

    print(f"Generating {args.num_samples} SVG(s)...")

    for i in range(args.num_samples):
        # Tokenize prompt
        inputs = tokenizer.apply_chat_template(
            PROMPT,
            return_tensors="pt",
            add_generation_prompt=True,
        )

        # Move to same device as model
        inputs = inputs.to(model.device)

        # Generate
        outputs = model.generate(
            inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract SVG
        svg_content = extract_svg(generated_text)

        # Save SVG
        output_file = output_dir / f"{checkpoint_name}_sample_{i}.svg"
        output_file.write_text(svg_content)
        print(f"  Saved: {output_file}")

        # Also print the SVG for inspection
        print(f"\n--- Sample {i} ---")
        print(svg_content[:500] + "..." if len(svg_content) > 500 else svg_content)
        print()


if __name__ == "__main__":
    main()
