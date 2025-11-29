import wandb
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import OnlineDPOConfig, OnlineDPOTrainer
from pairwise_judge import PairwiseJudge

from shared import PREFIX

MODEL_NAME = "Qwen/Qwen3-0.6B"
SYS_PROMPT = "You are an SVG generator. Respond only with valid SVG code."
CONSTANT_PROMPT = [
    {"role": "system", "content": SYS_PROMPT},
    {"role": "user", "content": "Generate SVG image of an apple"},
]
NUM_SAMPLES = 2
wandb.init(
    project="vequinox",
    name="apple-run-001", # TODO: change to document unique runs
    config={
        "base_model": MODEL_NAME,
        "guide_model": "claude-haiku-4-5-20251001",
        "num_samples": NUM_SAMPLES,
        "prompt": SYS_PROMPT,
    },
)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.chat_template = tokenizer.chat_template.replace("<|im_start|>assistant", "<|im_start|>assistant " + PREFIX)

judge = PairwiseJudge()

config = OnlineDPOConfig(
    output_dir="./vequinox-checkpoints",
    # wandb integration
    report_to="wandb",
    logging_steps=1,
    max_new_tokens=512,
    max_length=1024,
)


train_dataset = Dataset.from_dict({"prompt": [CONSTANT_PROMPT] * NUM_SAMPLES})

training_args = OnlineDPOConfig(
    output_dir="vequinox-checkpoints",
)

trainer = OnlineDPOTrainer(
    model=model,
    judge=judge,
    args=config,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)

print("=== Running training ===")
print("Device: ", trainer.accelerator.device)
judge.to(trainer.accelerator.device)
trainer.train()
