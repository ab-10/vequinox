# vequinox UI

The UI is a python file within the vequinox python project.
It allows users to load a model from `instruction-ft-checkpoints/` and interactively train it further.


## Stack

1. Streamlit

## Layout

The UI has 3 panes:
1. Interact
2. DPO
3. CLIP
4. Battle

### Interact

Load a model from a checkpoint.
User inputs the generation prompt into a textbox.
The generation prompt is passed to the model in a consistent format with `instruction_ft_eval.py`.
The model generates an SVG output.
The UI must construct the system prompt as `Generate an SVG image of the following object: {user_text}`, where `{user_text}` is the user's textbox input (replacing the `example["name"]` part used in `instruction_ft_eval.py`).
Before calling `apply_chat_template`, the UI must inject the same `PREFIX` from `shared.py` into the tokenizer chat template, by doing:

```
from shared import PREFIX

tokenizer.chat_template = tokenizer.chat_template.replace(
    "<|im_start|>assistant",
    "<|im_start|>assistant " + PREFIX,
)
    The model is then prompted using this templated chat format, and the decoded model output is treated as the SVG content (consistent with `instruction_ft_eval.py`).
The model generates an SVG output.
```

### DPO

Stub: out of scope

### CLIP

Stub: out of scope

### Battle

The user competes against image generation model.

1. Randomly select a prompt from our local `svgx_maxlen_1000` dataset's name column.
    1. Precompute the list as a local text file, that lists all name columns
2. Show to the user a prompt at the top
3. Display:
    1. The model's generation output on the left side
    2. A drawable box on the right side
    3. A submit button on the bottom
4. At submission send a rasterised image of model's generation and user's output to Claude Sonnet 4.5 API for scoring asking which image is more accurate.
5. Display an epic result at the end:
    1. If user wins: headline: "You win!", subtext "Did you just get lucky? Defend your title."
    2. If a user loses: headline: "What a loss!", subtext: "You came, you tried, you failed!"
    3. "Next Round" button on the bottom

## Roadmap

Outside of the scope.
Don't implement yet!

1. Battle mode