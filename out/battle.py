import os
import random
import base64
import io
import xml

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
        return "human", full_text
    if "MODEL" in full_text:
        return "model", full_text
    if "TIE" in full_text:
        return "tie", full_text
    return "unknown", full_text


def render_battle():
    from ui import get_model_and_tokenizer, generate_svgs

    st.header("Battle")
    checkpoints = list_checkpoints()
    checkpoint_names = [path.name for path in checkpoints]
    if not checkpoint_names:
        st.error("No checkpoints available.")
        return
    default_checkpoint_name = "checkpoint-80050"
    if default_checkpoint_name in checkpoint_names:
        default_checkpoint_name = default_checkpoint_name
    else:
        default_checkpoint_name = checkpoint_names[-1]
    selected_checkpoint_name = st.session_state.get(
        "checkpoint_name", default_checkpoint_name
    )
    if selected_checkpoint_name not in checkpoint_names:
        selected_checkpoint_name = default_checkpoint_name
    model, tokenizer, device, checkpoint = get_model_and_tokenizer(
        selected_checkpoint_name
    )
    prompts = load_battle_prompts()
    if "battle_prompt" not in st.session_state or st.session_state.get(
        "battle_new_round", False
    ):
        st.session_state["battle_prompt"] = random.choice(prompts) if prompts else ""
        st.session_state["battle_model_svg"] = None
        st.session_state["battle_result"] = None
        st.session_state["battle_new_round"] = False
    prompt_text = st.session_state["battle_prompt"]
    if not prompt_text:
        st.error("No prompts available for battle mode.")
        return
    if "battle_canvas_version" not in st.session_state:
        st.session_state["battle_canvas_version"] = 0
    st.subheader("Prompt")
    st.write(prompt_text)
    spinner_placeholder = st.empty()
    cols = st.columns(2, gap="large")
    with cols[0]:
        st.subheader("Model's image")
        model_image_placeholder = st.empty()
        model_image_placeholder.markdown(
            "<div style='height:512px'></div>",
            unsafe_allow_html=True,
        )
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
            key=f"battle_canvas_{st.session_state['battle_canvas_version']}",
        )
    if st.session_state["battle_model_svg"] is None:
        spinner_container = spinner_placeholder.container()
        with spinner_container:
            with st.spinner("Generating model image..."):
                svgs = generate_svgs(
                    model, tokenizer, device, prompt_text, num_images=1
                )
        if not svgs:
            st.error("Model did not generate any SVG.")
            return
        svg_prefix = PREFIX[PREFIX.index("<svg"):]
        full_svg = svg_prefix + svgs[0]
        try:
            load_svg_from_string(full_svg)
        except xml.etree.ElementTree.ParseError:
            st.error("Model generated an invalid SVG.")
            return
        st.session_state["battle_model_svg"] = full_svg
    if st.session_state["battle_model_svg"] is not None:
        model_image = load_svg_from_string(st.session_state["battle_model_svg"])
        model_image_placeholder.image(model_image, width=512)
    submit = st.button("Submit")
    if submit:
        if canvas_result.image_data is None:
            st.error("Please draw something before submitting.")
            return
        user_image_array = canvas_result.image_data
        user_image = Image.fromarray(user_image_array.astype("uint8"))
        model_image = load_svg_from_string(st.session_state["battle_model_svg"])
        with st.spinner("Asking Claude to judge the battle..."):
            outcome, raw_response = score_battle_with_claude(
                prompt_text, model_image, user_image
            )
        st.session_state["battle_result"] = (outcome, raw_response)
    if st.session_state.get("battle_result") is not None:
        outcome, _ = st.session_state["battle_result"]
        if outcome == "human":
            st.markdown("### You win!")
            st.write("Did you just get lucky? Defend your title.")
        elif outcome == "model":
            st.markdown("### What a loss!")
            st.write("You came, you tried, you failed!")
        elif outcome == "tie":
            st.markdown("### It's a tie!")
            st.write("So close. Go again and claim your victory.")
        else:
            st.markdown("### Unable to determine a winner")
            st.write("The judge could not clearly choose a winner. Try another round.")
        if st.button("Next Round"):
            st.session_state["battle_new_round"] = True
            st.session_state["battle_result"] = None
            st.session_state["battle_model_svg"] = None
            st.session_state["battle_canvas_version"] += 1
            st.rerun()


render_battle()

