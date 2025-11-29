import anthropic
from trl import BasePairwiseJudge

from scoring import load_clip, score_svg, load_svg_from_string
import torch
import re
import wandb
from shared import PREFIX
import base64
import io


class CLIPPairwiseJudge(BasePairwiseJudge):
    CLIP_JUDGE_PROMPT = """
Image of a pelican riding a bicycle.
The pelican should have a clear structure and look like a bird at least
The bicycle should be built up using a frame, wheels etc.
the pelican should be properly nested on the bicycle.
    """

    def __init__(self):
        self.clip_processor, self.clip_model = load_clip()

    def to(self, device: torch.device):
        self.clip_model = self.clip_model.to(device)

    @staticmethod
    def _extract_svg_from_completion(completion):
        # Extract content between triple backticks
        match = re.search(r"```\w*\n?(.*?)(?:```|$)", completion, re.DOTALL)
        return match.group(1) if match else None

    def postprocess_completion(self, completion):
        if completion.startswith("Assistant:"):
            print("Rogue 'Assistant:' prefix found in completion. Removing it.")
            print(completion)
            completion = completion.lstrip("Assistant:")
        return PREFIX + completion

    def judge(self, prompts, completions, shuffle_order=True):
        completions = [[self.postprocess_completion(a), self.postprocess_completion(b)] for (a, b) in completions]

        results = []
        for comp_a, comp_b in completions:
            # import pdb

            # pdb.set_trace()
            print(comp_a)
            svg_a = self._extract_svg_from_completion(comp_a)
            svg_b = self._extract_svg_from_completion(comp_b)
            if svg_a is None:
                print(f"could not extract svg from {comp_a}")
                results.append(-1)
                continue
            if svg_b is None:
                print(f"could not extract svg from {comp_b}")
                results.append(-1)
                continue
            score_dict = score_svg(
                [svg_a, svg_b],
                self.CLIP_JUDGE_PROMPT,
                self.clip_processor,
                self.clip_model,
            )
            results.append(int(score_dict["scores"].argmax().item()))
        return results


class ClaudePairwiseJudge(BasePairwiseJudge):
    """Pairwise judge that uses Claude's vision API to compare SVG candidates."""

    JUDGE_SYSTEM_PROMPT = """You are an expert SVG evaluator. Your task is to compare two SVG images and determine which one better matches the given prompt.

Evaluate based on the following criteria:
1. **Semantic Match**: How well does the SVG content align with the prompt's meaning?
2. **SVG Validity**: Is the SVG well-formed and renderable?
3. **Simplicity**: Are the shapes clean and not overly complex?
4. **Visual Quality**: Does the image look good and recognizable?

You must respond with ONLY a single character: either "A" or "B" to indicate which image is better.
If both images are equally good or equally bad, respond with "A".
If neither image can be evaluated (e.g., both are invalid), respond with "X"."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.model = model
        self.client = anthropic.Anthropic()

    def to(self, device: torch.device):
        pass

    @staticmethod
    def _extract_svg_from_completion(completion: str) -> str | None:
        match = re.search(r"```\w*\n?(.*?)(?:```|$)", completion, re.DOTALL)
        return match.group(1) if match else None

    def postprocess_completion(self, completion: str) -> str:
        if completion.startswith("Assistant:"):
            print("Rogue 'Assistant:' prefix found in completion. Removing it.")
            print(completion)
            completion = completion.lstrip("Assistant:")
        return PREFIX + completion

    def _svg_to_base64_png(self, svg_text: str) -> str | None:
        """Convert SVG text to base64-encoded PNG for Claude's vision API."""
        try:
            image = load_svg_from_string(svg_text)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            return base64.standard_b64encode(buffer.read()).decode("utf-8")
        except Exception as e:
            print(f"Error converting SVG to PNG: {e}")
            return None

    def _call_claude(self, prompt: str, image_a_b64: str, image_b_b64: str) -> str:
        """Call Claude API with two images and get the preference."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=10,
            system=self.JUDGE_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Prompt: {prompt}\n\nImage A:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_a_b64,
                            },
                        },
                        {"type": "text", "text": "Image B:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Which image better matches the prompt? Respond with only 'A' or 'B'.",
                        },
                    ],
                }
            ],
        )
        return message.content[0].text.strip()

    def judge(
        self, prompts: list[str], completions: list[list[str]], shuffle_order: bool = True
    ) -> list[int]:
        completions = [
            [self.postprocess_completion(a), self.postprocess_completion(b)]
            for (a, b) in completions
        ]

        results = []
        for i, (comp_a, comp_b) in enumerate(completions):
            prompt = prompts[i] if i < len(prompts) else prompts[0]

            svg_a = self._extract_svg_from_completion(comp_a)
            svg_b = self._extract_svg_from_completion(comp_b)

            if svg_a is None:
                print(f"Could not extract SVG from completion A: {comp_a[:100]}...")
                results.append(-1)
                continue
            if svg_b is None:
                print(f"Could not extract SVG from completion B: {comp_b[:100]}...")
                results.append(-1)
                continue

            image_a_b64 = self._svg_to_base64_png(svg_a)
            image_b_b64 = self._svg_to_base64_png(svg_b)

            if image_a_b64 is None or image_b_b64 is None:
                print("Could not convert one or both SVGs to images")
                results.append(-1)
                continue

            try:
                response = self._call_claude(prompt, image_a_b64, image_b_b64)
                wandb.log(
                    {
                        "claude_judge_response": response,
                        "svg_images": [
                            wandb.Image(load_svg_from_string(svg_a), caption="Image A"),
                            wandb.Image(load_svg_from_string(svg_b), caption="Image B"),
                        ],
                    }
                )

                if response.upper() == "A":
                    results.append(0)
                elif response.upper() == "B":
                    results.append(1)
                else:
                    print(f"Unexpected Claude response: {response}")
                    results.append(-1)
            except Exception as e:
                print(f"Error calling Claude API: {e}")
                results.append(-1)

        return results


# Backward compatibility alias
PairwiseJudge = CLIPPairwiseJudge


def create_judge(judge_type: str = "clip", model: str | None = None) -> BasePairwiseJudge:
    """Factory function to create a pairwise judge.

    Args:
        judge_type: Type of judge to create. Either "clip" or "claude".
        model: For Claude judge, the model to use. Defaults to "claude-haiku-4-5-20251001".

    Returns:
        A pairwise judge instance.
    """
    if judge_type.lower() == "clip":
        return CLIPPairwiseJudge()
    elif judge_type.lower() == "claude":
        model = model or "claude-haiku-4-5-20251001"
        return ClaudePairwiseJudge(model=model)
    else:
        raise ValueError(f"Unknown judge type: {judge_type}. Must be 'clip' or 'claude'.")
