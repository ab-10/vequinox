import io

import wandb
from datasets import load_from_disk, load_dataset
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import cairosvg

from shared import LOCAL_DATASET_PATH, MAX_SVG_LENGTH, PREFIX


MODEL_NAME = "Qwen/Qwen3-0.6B"
SYS_PROMPT = "You are an SVG generator. Respond only with valid SVG code."
NUM_EPOCHS = 50
DATASET_NAME = "xingxm/SVGX-Core-250k"


wandb.init(
    project="vequinox",
    config={
        "base_model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "num_epochs": NUM_EPOCHS,
    },
)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.chat_template = tokenizer.chat_template.replace(
    "<|im_start|>assistant", "<|im_start|>assistant " + PREFIX
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def svg_to_pil(svg_text, width=512, height=512):
    png_data = cairosvg.svg2png(
        bytestring=svg_text.encode("utf-8"),
        output_width=width,
        output_height=height,
    )
    return Image.open(io.BytesIO(png_data))



# ds = load_from_disk(LOCAL_DATASET_PATH)
ds = load_dataset(DATASET_NAME)

example = None
for potential_example in ds["train"]:
    if len(potential_example["svg_code"]) <= MAX_SVG_LENGTH:
        example = potential_example
        break

if example is None:
    raise ValueError(
        f"No example found with SVG code length less than or equal to {MAX_SVG_LENGTH}"
    )

target_svg = example["svg_code"]

SYS_PROMPT = "Generate an SVG image of the following object: " + example["name"]

messages_prompt = [
    {"role": "system", "content": SYS_PROMPT},
]

messages_full = messages_prompt + [
    {"role": "assistant", "content": target_svg},
]

prompt_ids = tokenizer.apply_chat_template(
    messages_prompt,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)

full_ids = tokenizer.apply_chat_template(
    messages_full,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt",
)

input_ids = full_ids[0]
labels = input_ids.clone()

prompt_length = prompt_ids.shape[1]
labels[:prompt_length] = -100

train_example = {
    "input_ids": input_ids,
    "labels": labels,
}


class SingleExampleDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return train_example


train_dataset = SingleExampleDataset()


training_args = TrainingArguments(
    output_dir="./instruction-ft-checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    logging_steps=1,
    save_steps=NUM_EPOCHS,
    report_to=["wandb"],
    learning_rate=1e-4,
    weight_decay=0.0,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

print("=== Running instruction fine-tuning on a single example ===")
print("Dataset:", DATASET_NAME)
print("System prompt:", SYS_PROMPT)
trainer.train()

model.eval()

with torch.no_grad():
    generated_ids = model.generate(
        prompt_ids.to(model.device),
        max_new_tokens=512,
    )

generated_tokens = generated_ids[0][prompt_length:]
generated_svg = tokenizer.decode(generated_tokens, skip_special_tokens=True)

try:
    target_image = svg_to_pil(target_svg)
    generated_image = svg_to_pil(generated_svg)
except Exception as e:
    print(f"Failed to render SVGs: {e}")
else:
    wandb.log(
        {
            "instruction_ft/target_svg": wandb.Image(
                target_image, caption="instruction_ft_target"
            ),
            "instruction_ft/generated_svg": wandb.Image(
                generated_image, caption="instruction_ft_generated"
            ),
        }
    )
