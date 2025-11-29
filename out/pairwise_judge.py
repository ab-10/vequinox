# import anthropic
from trl import BasePairwiseJudge

from scoring import load_clip, score_svg
import torch
import re
import wandb


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
        match = re.search(r"```\w*\n?(.*?)(?:\n?```)?$", completion, re.DOTALL)
        return match.group(1) if match else None

    def judge(self, prompts, completions, shuffle_order=True):
        results = []
        for _, (comp_a, comp_b) in zip(prompts, completions):
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
            result = score_svg(
                [svg_a, svg_b],
                self.CLIP_JUDGE_PROMPT,
                self.clip_processor,
                self.clip_model,
            )
            images, scores = result["images"], result["scores"]
            wandb.log({"svg_a": wandb.Image(images[0]), "svg_b": wandb.Image(images[1])})
            results.append(int(scores.argmax().item()))
        return results
