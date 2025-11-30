from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared import LOCAL_DATASET_PATH, MAX_SVG_LENGTH, PREFIX

CHECKPOINT_DIR = "instruction-ft-checkpoints"


def list_checkpoints():
    checkpoint_root = Path(CHECKPOINT_DIR)
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"No checkpoint directory found at {checkpoint_root}")
    checkpoints = [
        path
        for path in checkpoint_root.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint-")
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {checkpoint_root}")
    checkpoints.sort(key=lambda p: int(p.name.split("-")[-1]))
    return checkpoints


def get_eval_prompt():
    ds = load_from_disk(LOCAL_DATASET_PATH)
    example = None
    for potential_example in ds["train"]:
        if len(potential_example["svg_code"]) <= MAX_SVG_LENGTH:
            example = potential_example
    if example is None:
        raise ValueError(
            f"No example found with SVG code length less than or equal to {MAX_SVG_LENGTH}"
        )
    sys_prompt = "Generate an SVG image of the following object: " + example["name"]
    return sys_prompt


def load_model_and_tokenizer(selected_checkpoint_name=None):
    checkpoints = list_checkpoints()
    checkpoint = None
    if selected_checkpoint_name is not None:
        for path in checkpoints:
            if path.name == selected_checkpoint_name:
                checkpoint = path
                break
    if checkpoint is None:
        checkpoint = checkpoints[-1]
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return model, tokenizer, checkpoint


def main():
    sys_prompt = get_eval_prompt()
    model, tokenizer, last_checkpoint = load_model_and_tokenizer()

    tokenizer.chat_template = tokenizer.chat_template.replace(
        "<|im_start|>assistant",
        "<|im_start|>assistant " + PREFIX,
    )

    messages_prompt = [
        {"role": "system", "content": sys_prompt},
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompt_ids = tokenizer.apply_chat_template(
        messages_prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    prompt_ids = prompt_ids.to(device)
    prompt_length = prompt_ids.shape[1]

    with torch.no_grad():
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=512,
        )

    generated_tokens = generated_ids[0][prompt_length:]
    generated_svg = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print("Checkpoint:", str(last_checkpoint))
    print("System prompt:", sys_prompt)
    print("Generated SVG:")
    print(generated_svg)


if __name__ == "__main__":
    main()
