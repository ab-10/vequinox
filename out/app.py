import gradio as gr
import threading
import queue
import sys
from io import StringIO

# Training state
training_thread = None
training_logs = queue.Queue()
is_training = False


def run_training():
    """Run the training loop and capture output."""
    global is_training
    is_training = True
    
    # Capture stdout to get training logs
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    
    try:
        import wandb
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import OnlineDPOConfig, OnlineDPOTrainer
        from pairwise_judge import PairwiseJudge
        
        MODEL_NAME = "Qwen/Qwen3-0.6B"
        PROMPT = """Generate an SVG of a pelican riding a bicycle"""
        CONSTANT_PROMPT = [{"role": "user", "content": PROMPT}]
        NUM_SAMPLES = 10
        
        training_logs.put("Initializing wandb...")
        wandb.init(
            project="vequinox",
            config={
                "base_model": MODEL_NAME,
                "guide_model": "claude-haiku-4-5-20251001",
                "num_samples": NUM_SAMPLES,
                "prompt": PROMPT,
            }
        )
        
        training_logs.put(f"Loading model: {MODEL_NAME}...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        training_logs.put("Initializing judge...")
        judge = PairwiseJudge()
        
        training_logs.put("Creating dataset...")
        train_dataset = Dataset.from_dict({"prompt": [CONSTANT_PROMPT] * NUM_SAMPLES})
        
        training_logs.put("Configuring trainer...")
        config = OnlineDPOConfig(
            output_dir="./vequinox-checkpoints",
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
        
        training_logs.put(f"=== Running training ===")
        training_logs.put(f"Device: {trainer.accelerator.device}")
        
        trainer.train()
        
        training_logs.put("Training complete!")
        wandb.finish()
        
    except Exception as e:
        training_logs.put(f"Error: {str(e)}")
    finally:
        # Restore stdout
        sys.stdout = old_stdout
        is_training = False


def start_training():
    """Start training in a background thread."""
    global training_thread
    
    if is_training:
        return "Training already in progress..."
    
    # Clear the queue
    while not training_logs.empty():
        training_logs.get()
    
    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()
    
    return "Training started! Check logs below..."


def get_logs():
    """Get accumulated training logs."""
    logs = []
    while not training_logs.empty():
        try:
            logs.append(training_logs.get_nowait())
        except queue.Empty:
            break
    return "\n".join(logs) if logs else "No new logs..."


def get_status():
    """Get current training status."""
    if is_training:
        return "🟢 Training in progress..."
    return "⚪ Idle"


# Build Gradio interface
with gr.Blocks(title="vequinox - SVG Training") as demo:
    gr.Markdown("# 🦅 vequinox")
    gr.Markdown("Post-train an LLM to generate SVGs using RL feedback from a larger model")
    
    with gr.Row():
        with gr.Column():
            status_text = gr.Textbox(label="Status", value="⚪ Idle", interactive=False)
            train_btn = gr.Button("🚀 Start Training Loop", variant="primary", size="lg")
            refresh_btn = gr.Button("🔄 Refresh Logs")
        
        with gr.Column():
            gr.Markdown("### Training Configuration")
            gr.Markdown("""
            - **Base Model:** Qwen/Qwen3-0.6B
            - **Guide Model:** Claude Haiku 4.5
            - **Prompt:** Generate an SVG of a pelican riding a bicycle
            - **Samples:** 10
            """)
    
    logs_output = gr.Textbox(
        label="Training Logs",
        lines=15,
        max_lines=30,
        interactive=False,
        placeholder="Training logs will appear here..."
    )
    
    gr.Markdown("---")
    gr.Markdown("📊 View detailed metrics on [Weights & Biases](https://wandb.ai)")
    
    # Event handlers
    train_btn.click(fn=start_training, outputs=logs_output)
    refresh_btn.click(fn=get_logs, outputs=logs_output)
    refresh_btn.click(fn=get_status, outputs=status_text)


if __name__ == "__main__":
    demo.launch()
