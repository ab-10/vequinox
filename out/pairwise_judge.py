# import anthropic
from trl import BasePairwiseJudge

from scoring import load_clip, score_images, svg_text_to_pil
import torch
import re


class PairwiseJudge(BasePairwiseJudge):
    CLIP_JUDGE_PROMPT = """
image of a pelican riding a bicycle.
The pelican should have a clear structure and look like a bird atleast
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

    def _preprocess_completion(self, completion, width, height):
        """Extract SVG from completion and convert to image.

        Returns:
            PIL Image on success, None on failure (logs the error)
        """
        svg = self._extract_svg_from_completion(completion)
        if svg is None:
            print(f"could not extract svg from {completion}")
            return None
        try:
            img = svg_text_to_pil(svg, width=width, height=height)
        except Exception as e:
            print(f"failed to convert svg to image: {e}")
            return None
        if img is None:
            print("svg_text_to_pil returned None")
            return None
        return img

    def judge(self, prompts, completions, shuffle_order=True):
        width = self.clip_processor.image_processor.crop_size["width"]
        height = self.clip_processor.image_processor.crop_size["height"]

        results = []
        for _, (comp_a, comp_b) in zip(prompts, completions):
            img_a = self._preprocess_completion(comp_a, width, height)
            img_b = self._preprocess_completion(comp_b, width, height)
            if img_a is None or img_b is None:
                results.append(-1)
                continue

            scores = score_images(
                [img_a, img_b],
                self.CLIP_JUDGE_PROMPT,
                self.clip_processor,
                self.clip_model,
            )
            results.append(int(scores.argmax().item()))
        return results
