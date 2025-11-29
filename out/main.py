import os
import random
import anthropic
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import OnlineDPOConfig, OnlineDPOTrainer


from pairwise_judge import PairwiseJudge

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

judge = PairwiseJudge()

CONSTANT_PROMPT = [
    {"role": "user", "content": "Write a short poem about the moon."}
]
NUM_SAMPLES = 1000

train_dataset = Dataset.from_dict({"prompt": [CONSTANT_PROMPT] * NUM_SAMPLES})

training_args = OnlineDPOConfig(output_dir="Qwen2-0.5B-OnlineDPO-Claude")

trainer = OnlineDPOTrainer(
    model=model,
    judge=judge,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)

print("=== Running training ===")
print("Device: ", trainer.accelerator.device)

trainer.train()
