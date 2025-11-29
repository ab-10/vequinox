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
DATASET_NAME = LOCAL_DATASET_PATH


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


ds = load_from_disk(LOCAL_DATASET_PATH)
# ds = load_dataset(DATASET_NAME)


class InstructionDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        sys_prompt = "Generate an SVG image of the following object: " + example["name"]

        messages_prompt = [
            {"role": "system", "content": sys_prompt},
        ]

        messages_full = messages_prompt + [
            {"role": "assistant", "content": example["svg_code"]},
        ]

        prompt_ids_local = tokenizer.apply_chat_template(
            messages_prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        full_ids_local = tokenizer.apply_chat_template(
            messages_full,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )

        input_ids = full_ids_local[0]
        labels = input_ids.clone()

        prompt_length_local = prompt_ids_local.shape[1]
        labels[:prompt_length_local] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


train_dataset = InstructionDataset(ds["train"])

if len(train_dataset) == 0:
    raise ValueError(
        f"No examples found in the train split at {LOCAL_DATASET_PATH}"
    )

example = ds["train"][0]
target_svg = example["svg_code"]
eval_sys_prompt = "Generate an SVG image of the following object: " + example["name"]

messages_prompt = [
    {"role": "system", "content": eval_sys_prompt},
]

prompt_ids = tokenizer.apply_chat_template(
    messages_prompt,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)

prompt_length = prompt_ids.shape[1]


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

print("=== Running instruction fine-tuning on full train split ===")
print("Number of training examples:", len(train_dataset))
print("Dataset:", DATASET_NAME)
print("System prompt template:", SYS_PROMPT)
trainer.train()

model.eval()

with torch.no_grad():
    generated_ids = model.generate(
        prompt_ids.to(model.device),
        max_new_tokens=512,
    )

generated_tokens = generated_ids[0][prompt_length:]
generated_svg = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("Sample generated SVG:")
print(generated_svg)

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
