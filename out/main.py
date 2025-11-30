import argparse

import wandb
from datasets import Dataset

# from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import OnlineDPOConfig, OnlineDPOTrainer
from pairwise_judge import PairwiseJudge

from shared import MODEL_NAME, PREFIX, PROMPT, SYS_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(description="Train vequinox model")
    parser.add_argument(
        "--num_train_epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--num_samples", type=int, default=500, help="Number of training samples"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./vequinox-checkpoints",
        help="Output directory for checkpoints",
    )
    return parser.parse_args()


args = parse_args()

NUM_SAMPLES = args.num_samples
# LORA_R = 32
# LORA_ALPHA = 64
# LORA_DROPOUT = 0.05
# LORA_TARGET_MODULES = [
#     "q_proj",
#     "k_proj",
#     "v_proj",
#     "o_proj",
#     "gate_proj",
#     "up_proj",
#     "down_proj",
# ]
#
# peft_config = LoraConfig(
#     r=LORA_R,
#     lora_alpha=LORA_ALPHA,
#     lora_dropout=LORA_DROPOUT,
#     bias="none",
#     target_modules=LORA_TARGET_MODULES,
#     task_type="CAUSAL_LM",
# )

wandb.init(
    project="vequinox",
    # name="pelican-run-001", # TODO: change to document unique runs
    config={
        "base_model": MODEL_NAME,
        "guide_model": "claude-haiku-4-5-20251001",
        "num_samples": NUM_SAMPLES,
        "prompt": SYS_PROMPT,
        # "lora_r": LORA_R,
        # "lora_alpha": LORA_ALPHA,
        # "lora_dropout": LORA_DROPOUT,
        # "lora_target_modules": LORA_TARGET_MODULES,
    },
)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.chat_template = tokenizer.chat_template.replace(
    "<|im_start|>assistant", "<|im_start|>assistant " + PREFIX
)

judge = PairwiseJudge()

config = OnlineDPOConfig(
    output_dir=args.output_dir,
    # wandb integration
    report_to="wandb",
    logging_steps=1,
    max_new_tokens=512,
    max_length=1024,
    save_steps=20,
    save_total_limit=5,
    num_train_epochs=args.num_train_epochs,
)


train_dataset = Dataset.from_dict({"prompt": [PROMPT] * NUM_SAMPLES})

trainer = OnlineDPOTrainer(
    model=model,
    judge=judge,
    args=config,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    # peft_config=peft_config,
)

print("=== Running training ===")
print("Device: ", trainer.accelerator.device)
judge.to(trainer.accelerator.device)
trainer.train()
