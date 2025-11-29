import os
import wandb
import random
import anthropic
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import OnlineDPOConfig, OnlineDPOTrainer


from pairwise_judge import PairwiseJudge

MODEL_NAME = "Qwen/Qwen3-0.6B"

PROMPT = """Generate an SVG of a pelican riding a bicycle"""
CONSTANT_PROMPT = [
    {"role": "user", "content": PROMPT}
]
NUM_SAMPLES = 10

wandb.init(
    project="vequinox",
    # name="pelican-run-001", # TODO: change to document unique runs
    config={
        "base_model": MODEL_NAME,
        "guide_model": "claude-haiku-4-5-20251001",
        "num_samples": NUM_SAMPLES,
        "prompt": PROMPT,
    }
)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

judge = PairwiseJudge()



train_dataset = Dataset.from_dict({"prompt": [CONSTANT_PROMPT] * NUM_SAMPLES})

config = OnlineDPOConfig(
    output_dir="./vequinox-checkpoints",
    
    # wandb integration
    report_to="wandb",
    logging_steps=1,
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

trainer.train()
