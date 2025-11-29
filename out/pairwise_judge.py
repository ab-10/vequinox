# import anthropic
from trl import BasePairwiseJudge

from scoring import load_clip, score_svg
import torch
import re
import wandb
from shared import PREFIX


class PairwiseJudge(BasePairwiseJudge):
    CLIP_JUDGE_PROMPT = """
Image of an apple.
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
