import os
import random
import base64
import io
import xml.etree.ElementTree

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from datasets import load_from_disk
from PIL import Image
import anthropic

from instruction_ft_eval import list_checkpoints
from scoring import load_svg_from_string
from shared import PREFIX, LOCAL_DATASET_PATH


@st.cache_resource
def load_battle_prompts():
    names_path = os.path.join(LOCAL_DATASET_PATH, "names.txt")
    if os.path.exists(names_path):
        with open(names_path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        if names:
            return names
    ds = load_from_disk(LOCAL_DATASET_PATH)
    train_ds = ds["train"]
    names = [example["name"] for example in train_ds]
    os.makedirs(LOCAL_DATASET_PATH, exist_ok=True)
    with open(names_path, "w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")
    return names


def load_anthropic_api_key():
    base_dir = os.path.dirname(__file__)
    env_path = os.path.join(base_dir, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("ANTHROPIC_API_KEY"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        return parts[1].strip().strip('"').strip("'")
    return os.environ.get("ANTHROPIC_API_KEY")


@st.cache_resource
def get_anthropic_client():
    api_key = load_anthropic_api_key()
    if api_key:
        return anthropic.Anthropic(api_key=api_key)
    return anthropic.Anthropic()


def pil_to_base64_png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def score_battle_with_claude(prompt_text, model_image, user_image):
    client = get_anthropic_client()
    model_b64 = pil_to_base64_png(model_image)
    user_b64 = pil_to_base64_png(user_image)
    content = [
        {
            "type": "text",
            "text": (
                "You are judging a drawing battle between an image generation model "
                "and a human user. The task is to draw an image that best matches "
                "the given prompt. You will be shown two images.\n\n"
                f"Prompt: {prompt_text}\n\n"
                "Image A is the model's image. Image B is the human's image.\n\n"
                "Respond with exactly one word: 'MODEL' if Image A is more accurate, "
                "'HUMAN' if Image B is more accurate, or 'TIE' if they are equally accurate."
            ),
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": model_b64,
            },
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": user_b64,
            },
        },
    ]
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=16,
        messages=[{"role": "user", "content": content}],
    )
    text_blocks = [block.text for block in response.content if block.type == "text"]
    full_text = " ".join(text_blocks).strip().upper()
    if "HUMAN" in full_text:
        return "human"
    if "MODEL" in full_text:
        return "model"
    if "TIE" in full_text:
        return "tie"
    return "unknown"


@st.cache_data
def generate_model_svg(prompt_text, checkpoint_name):
    """Generate SVG for a given prompt. Cached to prevent regeneration."""
    from ui import get_model_and_tokenizer, generate_svgs
    
    model, tokenizer, device, _ = get_model_and_tokenizer(checkpoint_name)
    svgs = generate_svgs(model, tokenizer, device, prompt_text, num_images=1)
    
    if not svgs:
        return None
    
    svg_prefix = PREFIX[PREFIX.index("<svg"):]
    full_svg = svg_prefix + svgs[0]
    
    try:
        load_svg_from_string(full_svg)
        return full_svg
    except xml.etree.ElementTree.ParseError:
        return None


def render_battle():
    st.header("Battle")
    
    # Get checkpoint
    checkpoints = list_checkpoints()
    checkpoint_names = [path.name for path in checkpoints]
    if not checkpoint_names:
        st.error("No checkpoints available.")
        return

    default_checkpoint_name = "checkpoint-80050"
    if default_checkpoint_name not in checkpoint_names:
        default_checkpoint_name = checkpoint_names[-1]
    checkpoint_name = default_checkpoint_name

    # Get prompts
    prompts = load_battle_prompts()
    if not prompts:
        st.error("No prompts available for battle mode.")
        return

    # Initialize prompt once per session
    if "battle_prompt" not in st.session_state:
        st.session_state["battle_prompt"] = random.choice(prompts)
    
    prompt_text = st.session_state["battle_prompt"]
    
    st.subheader("Prompt")
    st.write(prompt_text)

    # Check if we have a result to show
    if st.session_state.get("battle_result") is not None:
        result = st.session_state["battle_result"]
        outcome = result["outcome"]
        
        st.divider()
        
        cols = st.columns(2, gap="large")
        with cols[0]:
            st.subheader("Model's image")
            st.image(result["model_image"], width=512)
        with cols[1]:
            st.subheader("Your drawing")
            st.image(result["user_image"], width=512)
        
        st.divider()
        
        if outcome == "human":
            st.markdown("## 🎉 You win!")
            st.write("Did you just get lucky? Refresh the page to defend your title.")
        elif outcome == "model":
            st.markdown("## 🤖 What a loss!")
            st.write("You came, you tried, you failed! Refresh to try again.")
        elif outcome == "tie":
            st.markdown("## 🤝 It's a tie!")
            st.write("So close. Refresh the page and claim your victory.")
        else:
            st.markdown("## ❓ Unable to determine a winner")
            st.write("The judge could not clearly choose a winner. Refresh to try again.")
        
        return

    # Generate model image (cached by prompt and checkpoint)
    with st.spinner("Generating model image..."):
        model_svg = generate_model_svg(prompt_text, checkpoint_name)
    
    if model_svg is None:
        st.error("Model failed to generate a valid SVG. Refresh to try again.")
        return

    model_image = load_svg_from_string(model_svg)

    # Show the battle interface
    cols = st.columns(2, gap="large")
    with cols[0]:
        st.subheader("Model's image")
        st.image(model_image, width=512)
    with cols[1]:
        st.subheader("Your drawing")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=3,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=512,
            width=512,
            drawing_mode="freedraw",
            key="battle_canvas",
        )

    if st.button("Judge the Battle", type="primary"):
        if canvas_result.image_data is None:
            st.error("Please draw something before submitting.")
            return
        
        user_image = Image.fromarray(canvas_result.image_data.astype("uint8"))
        
        with st.spinner("Asking Claude to judge the battle..."):
            outcome = score_battle_with_claude(prompt_text, model_image, user_image)
        
        # Store result and rerun to show results view
        st.session_state["battle_result"] = {
            "outcome": outcome,
            "model_image": model_image,
            "user_image": user_image,
        }
        st.rerun()


render_battle()