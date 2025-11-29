import wandb
from datasets import Dataset
from peft import LoraConfig
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
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

wandb.init(
    project="vequinox",
    name="apple-run-001", # TODO: change to document unique runs
    config={
        "base_model": MODEL_NAME,
        "guide_model": "claude-haiku-4-5-20251001",
        "num_samples": NUM_SAMPLES,
        "prompt": SYS_PROMPT,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGET_MODULES,
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
    logging_steps=5,
    max_new_tokens=512,
    max_length=1024,
)

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=LORA_TARGET_MODULES,
    task_type="CAUSAL_LM",
)

train_dataset = Dataset.from_dict({"prompt": [CONSTANT_PROMPT] * NUM_SAMPLES})

trainer = OnlineDPOTrainer(
    model=model,
    judge=judge,
    args=config,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    peft_config=peft_config,
)

print("=== Running training ===")
print("Device: ", trainer.accelerator.device)
judge.to(trainer.accelerator.device)
trainer.train()
