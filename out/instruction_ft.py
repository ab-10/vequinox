import wandb
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

from shared import PREFIX


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

ds = load_dataset(DATASET_NAME)

example = None
for potential_example in ds["train"]:
    if len(example["svg_code"]) < 1000:
        example = potential_example

if example is None:
    raise ValueError("No example found with SVG code length less than 1000")

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
    torch_dtype=torch.bfloat16
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

