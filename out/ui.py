import xml
import time
import os

import torch
import streamlit as st
import streamlit.components.v1 as components

from instruction_ft_eval import load_model_and_tokenizer, list_checkpoints
from scoring import load_svg_from_string
from shared import PREFIX, MAX_SVG_LENGTH


@st.cache_resource
def get_model_and_tokenizer(selected_checkpoint_name):
    model, tokenizer, last_checkpoint = load_model_and_tokenizer(
        selected_checkpoint_name
    )
    tokenizer.chat_template = tokenizer.chat_template.replace(
        "<|im_start|>assistant",
        "<|im_start|>assistant " + PREFIX,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    return model, tokenizer, device, str(last_checkpoint)


def generate_svgs(model, tokenizer, device, prompt_text, num_images=4):
    t0 = time.time()
    sys_prompt = "Generate an SVG image of the following object: " + prompt_text
    messages_prompt = [
        {"role": "system", "content": sys_prompt},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages_prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    prompt_ids = prompt_ids.to(device)
    t1 = time.time()
    print(f"[generate_svgs] prompt prep took {t1 - t0:.3f}s")
    prompt_length = prompt_ids.shape[1]
    with torch.no_grad():
        t2 = time.time()
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=MAX_SVG_LENGTH,
            num_return_sequences=num_images,
            do_sample=True,
        )
        t3 = time.time()
    print(f"[generate_svgs] model.generate took {t3 - t2:.3f}s")
    svgs = []
    t4 = time.time()
    for i in range(num_images):
        generated_tokens = generated_ids[i][prompt_length:]
        generated_svg = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        svgs.append(generated_svg)
    t5 = time.time()
    print(f"[generate_svgs] decoding {num_images} sequences took {t5 - t4:.3f}s")
    print(f"[generate_svgs] total generate_svgs time {t5 - t0:.3f}s")
    return svgs


def render_dpo():
    st.header("DPO")
    t0 = time.time()
    checkpoints = list_checkpoints()
    checkpoint_names = [path.name for path in checkpoints]
    if not checkpoint_names:
        st.error("No checkpoints available.")
        return
    default_checkpoint_name = "checkpoint-80050"
    if default_checkpoint_name in checkpoint_names:
        default_index = checkpoint_names.index(default_checkpoint_name)
    else:
        default_index = len(checkpoint_names) - 1
    with st.expander("Configs for the curious"):
        selected_checkpoint_name = st.selectbox(
            "Checkpoint",
            checkpoint_names,
            index=default_index,
            key="dpo_checkpoint_name",
        )
    model, tokenizer, device, checkpoint = get_model_and_tokenizer(
        selected_checkpoint_name
    )
    t1 = time.time()
    print(f"[render_dpo] get_model_and_tokenizer took {t1 - t0:.3f}s")
    st.write("What do you want to generate?")
    user_text = st.text_input(
        "Enter a prompt for the SVG generator",
        key="dpo_prompt",
    )
    if "dpo_feedback" not in st.session_state:
        st.session_state["dpo_feedback"] = []
    if "dpo_pair" not in st.session_state:
        st.session_state["dpo_pair"] = None
    generate_pair = False
    button_label = "Start judging" if st.session_state["dpo_pair"] is None else "Next pair"
    if st.button(button_label, disabled=not bool(user_text)):
        generate_pair = True
    if generate_pair and user_text:
        t2 = time.time()
        print(
            f"[render_dpo] starting pair generation for prompt length {len(user_text)}"
        )
        with st.spinner("Generating SVG pair..."):
            svgs = generate_svgs(model, tokenizer, device, user_text, num_images=2)
        t3 = time.time()
        print(f"[render_dpo] generate_svgs took {t3 - t2:.3f}s")
        svg_prefix = PREFIX[PREFIX.index("<svg"):]
        full_svgs = []
        for svg in svgs:
            full_svg = svg_prefix + svg
            try:
                load_svg_from_string(full_svg)
            except xml.etree.ElementTree.ParseError:
                continue
            full_svgs.append(full_svg)
        if len(full_svgs) < 2:
            st.error("Could not generate two valid SVGs for this prompt. Try again.")
            st.session_state["dpo_pair"] = None
        else:
            st.session_state["dpo_pair"] = full_svgs[:2]
    pair = st.session_state.get("dpo_pair")
    if pair:
        left_svg, right_svg = pair
        cols = st.columns(2, gap="large")
        with cols[0]:
            st.subheader("Option A")
            components.html(left_svg, height=800)
            prefer_left = st.button("Prefer A", key="dpo_prefer_left")
        with cols[1]:
            st.subheader("Option B")
            components.html(right_svg, height=800)
            prefer_right = st.button("Prefer B", key="dpo_prefer_right")
        if prefer_left or prefer_right:
            preferred = 0 if prefer_left else 1
            st.session_state["dpo_feedback"].append(
                {
                    "prompt": user_text,
                    "svg_a": left_svg,
                    "svg_b": right_svg,
                    "preferred": preferred,
                    "timestamp": time.time(),
                }
            )
            st.session_state["dpo_pair"] = None
            st.rerun()
    if st.session_state["dpo_feedback"]:
        st.write(
            f"Collected {len(st.session_state['dpo_feedback'])} preference judgments so far."
        )


def render_interact():
    st.header("Interact")
    t0 = time.time()
    checkpoints = list_checkpoints()
    checkpoint_names = [path.name for path in checkpoints]
    if not checkpoint_names:
        st.error("No checkpoints available.")
        return
    default_checkpoint_name = "checkpoint-80050"
    if default_checkpoint_name in checkpoint_names:
        default_index = checkpoint_names.index(default_checkpoint_name)
    else:
        default_index = len(checkpoint_names) - 1
    with st.expander("Configs for the curious"):
        selected_checkpoint_name = st.selectbox(
            "Checkpoint",
            checkpoint_names,
            index=default_index,
            key="checkpoint_name",
        )
    model, tokenizer, device, checkpoint = get_model_and_tokenizer(
        selected_checkpoint_name
    )
    t1 = time.time()
    print(f"[render_interact] get_model_and_tokenizer took {t1 - t0:.3f}s")
    st.write("What do you want to generate?")
    user_text = st.text_input("Icon descriptions and simple objects work best!")

    if "valid_svgs" not in st.session_state:
        st.session_state["valid_svgs"] = None
    run_generation = False
    if st.button("Generate SVG") and user_text:
        run_generation = True
    if st.session_state["valid_svgs"] is not None and not st.session_state["valid_svgs"]:
        st.write("No valid images generated, retry?")
        if st.button("Retry generation") and user_text:
            run_generation = True
    if run_generation and user_text:
        t2 = time.time()
        print(
            f"[render_interact] starting generation for prompt length {len(user_text)}"
        )
        with st.spinner("Generating SVG..."):
            svgs = generate_svgs(model, tokenizer, device, user_text, num_images=4)
        t3 = time.time()
        print(f"[render_interact] generate_svgs took {t3 - t2:.3f}s")
        svg_prefix = PREFIX[PREFIX.index("<svg"):]
        full_svgs = [svg_prefix + svg for svg in svgs]
        t4 = time.time()
        valid_svgs = []
        for full_svg in full_svgs:
            try:
                load_svg_from_string(full_svg)
            except xml.etree.ElementTree.ParseError:
                continue
            valid_svgs.append(full_svg)
        t5 = time.time()
        print(
            "[render_interact] validation took "
            f"{t5 - t4:.3f}s, valid={len(valid_svgs)}/{len(full_svgs)}"
        )
        st.session_state["valid_svgs"] = valid_svgs
        if not valid_svgs:
            st.error("No valid SVGs were generated.")
        t6 = time.time()
        print(f"[render_interact] rendering took {t6 - t5:.3f}s")
        print(f"[render_interact] total after button click {t6 - t2:.3f}s")
    valid_svgs = st.session_state["valid_svgs"]
    if valid_svgs:
        full_svg = valid_svgs[0]
        components.html(full_svg, height=800)
        st.subheader("SVG code")
        st.code(full_svg, language="xml")


def render_stub(name):
    st.header(name)
    st.write("Not implemented yet.")


def main():
    st.set_page_config(page_title="vequinox UI", layout="wide")
    base_dir = os.path.dirname(__file__)
    video_path = os.path.join(base_dir, "assets", "pelican.mp4")
    st.title("vequinox")
    st.text("The equinox for vector graphics generation.")
    st.video(video_path, autoplay=True, muted=True, loop=True)
    st.page_link("battle.py", label="Battle mode")
    tab_interact, tab_dpo, tab_clip = st.tabs(["Interact", "DPO", "CLIP"])
    with tab_interact:
        render_interact()
    with tab_dpo:
        render_dpo()
    with tab_clip:
        render_stub("CLIP")


if __name__ == "__main__":
    main()

