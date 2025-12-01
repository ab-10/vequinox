"""
GEPA (Genetic-Pareto) inspired training for SVG generation.

This implements a simplified version of the GEPA approach from the paper
"GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning".

Key differences from the current Online DPO approach:
- Instead of updating model weights via DPO, we evolve the system prompt
- Uses natural language reflection to propose prompt improvements
- Maintains a pool of candidate prompts and selects based on performance
- Model weights can optionally be fine-tuned with LoRA separately

The core idea: Use an LLM to reflect on what makes good/bad SVGs and
iteratively improve the prompt based on this feedback.
"""

import random
import xml
from dataclasses import dataclass, field

import anthropic
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from scoring import load_clip, load_svg_from_string, score_svg
from shared import PREFIX

MODEL_NAME = "Qwen/Qwen3-0.6B"
REFLECTION_MODEL = "claude-haiku-4-5-20251001"

BASE_SYSTEM_PROMPT = "You are an SVG generator. Respond only with valid SVG code. /no_think"

USER_PROMPT = "Generate SVG code of a pelican riding a bicycle"

CLIP_EVAL_PROMPT = """
Image of a pelican riding a bicycle.
The pelican should have a clear structure and look like a bird at least.
The bicycle should be built up using a frame, wheels etc.
The pelican should be properly nested on the bicycle.
"""


@dataclass
class PromptCandidate:
    system_prompt: str
    scores: list[float] = field(default_factory=list)
    generation: int = 0
    parent_idx: int | None = None

    @property
    def avg_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)


@dataclass
class GEPAConfig:
    num_iterations: int = 10
    samples_per_iteration: int = 4
    pool_size: int = 5
    minibatch_size: int = 2
    improvement_threshold: float = 0.5
    max_new_tokens: int = 512
    temperature: float = 0.7


class GEPATrainer:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: GEPAConfig,
        device: torch.device | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

        self.clip_processor, self.clip_model = load_clip()
        self.clip_model.to(self.device)

        self.anthropic_client = anthropic.Anthropic()

        self.prompt_pool: list[PromptCandidate] = []

        self.tokenizer.chat_template = self.tokenizer.chat_template.replace(
            "<|im_start|>assistant", "<|im_start|>assistant " + PREFIX
        )

    def _extract_svg(self, completion: str) -> str | None:
        import re

        patterns = [
            r"```\w*\n?(.*?)(?:```|$)",
            r"(<svg.*?</svg>)",
        ]
        for pattern in patterns:
            match = re.search(pattern, completion, re.DOTALL)
            if match:
                return match.group(1)
        return None

    def generate_svg(self, system_prompt: str, user_prompt: str) -> tuple[str, str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = full_output[len(input_text) :].strip()

        svg_text = self._extract_svg(PREFIX + completion)
        if svg_text is None:
            svg_text = PREFIX + completion

        return svg_text, completion

    def score_single_svg(self, svg_text: str) -> float:
        try:
            score_dict = score_svg(
                [svg_text], CLIP_EVAL_PROMPT, self.clip_processor, self.clip_model
            )
            return float(score_dict["scores"][0])
        except (xml.etree.ElementTree.ParseError, Exception) as e:
            print(f"Error scoring SVG: {e}")
            return 0.0

    def generate_and_score_batch(
        self, system_prompt: str, user_prompt: str, n_samples: int
    ) -> list[tuple[str, str, float]]:
        results = []
        for _ in range(n_samples):
            svg_text, completion = self.generate_svg(system_prompt, user_prompt)
            score = self.score_single_svg(svg_text)
            results.append((svg_text, completion, score))
        return results

    def reflect_and_propose(
        self,
        current_prompt: str,
        samples_with_scores: list[tuple[str, str, float]],
        user_prompt: str,
    ) -> str:
        sorted_samples = sorted(samples_with_scores, key=lambda x: x[2], reverse=True)

        best_samples = sorted_samples[: min(2, len(sorted_samples))]
        worst_samples = sorted_samples[-min(2, len(sorted_samples)) :]

        reflection_prompt = f"""You are an expert at optimizing prompts for SVG generation models.

Current system prompt being used:
<current_prompt>
{current_prompt}
</current_prompt>

The user asked the model to: "{user_prompt}"

Here are some generated SVGs and their quality scores (higher is better):

BEST PERFORMING SVGs:
{self._format_samples(best_samples)}

WORST PERFORMING SVGs:
{self._format_samples(worst_samples)}

Analyze what makes the good SVGs better than the bad ones. Consider:
1. SVG structure and validity
2. Visual clarity and recognizable shapes
3. Proper composition (pelican on bicycle)
4. Clean, simple code vs overly complex code

Based on your analysis, propose an IMPROVED system prompt that will help the model generate better SVGs.
The new prompt should:
- Be clear and specific about what makes a good SVG
- Include guidance on structure, shapes, and composition
- Encourage valid SVG syntax
- Be concise but comprehensive

Respond with ONLY the new system prompt, nothing else. Do not include any explanation or preamble."""

        response = self.anthropic_client.messages.create(
            model=REFLECTION_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": reflection_prompt}],
        )

        new_prompt = response.content[0].text.strip()
        return new_prompt

    def _format_samples(self, samples: list[tuple[str, str, float]]) -> str:
        formatted = []
        for i, (svg_text, completion, score) in enumerate(samples):
            truncated_svg = svg_text[:500] + "..." if len(svg_text) > 500 else svg_text
            formatted.append(f"Sample {i+1} (score: {score:.2f}):\n{truncated_svg}\n")
        return "\n".join(formatted)

    def select_candidate(self) -> int:
        if not self.prompt_pool:
            return -1

        scores = [c.avg_score for c in self.prompt_pool]
        min_score = min(scores)
        adjusted_scores = [s - min_score + 1.0 for s in scores]
        total = sum(adjusted_scores)
        probs = [s / total for s in adjusted_scores]

        return random.choices(range(len(self.prompt_pool)), weights=probs, k=1)[0]

    def prune_pool(self):
        if len(self.prompt_pool) > self.config.pool_size:
            self.prompt_pool.sort(key=lambda c: c.avg_score, reverse=True)
            self.prompt_pool = self.prompt_pool[: self.config.pool_size]

    def train(self, user_prompt: str = USER_PROMPT) -> PromptCandidate:
        print("=== Starting GEPA Training ===")
        print(f"Device: {self.device}")
        print(f"Config: {self.config}")

        initial_candidate = PromptCandidate(
            system_prompt=BASE_SYSTEM_PROMPT, generation=0
        )
        initial_samples = self.generate_and_score_batch(
            BASE_SYSTEM_PROMPT, user_prompt, self.config.samples_per_iteration
        )
        initial_candidate.scores = [s[2] for s in initial_samples]
        self.prompt_pool.append(initial_candidate)

        print(f"Initial prompt avg score: {initial_candidate.avg_score:.2f}")

        wandb.log(
            {
                "iteration": 0,
                "best_avg_score": initial_candidate.avg_score,
                "pool_size": len(self.prompt_pool),
                "generation": 0,
            }
        )

        for iteration in range(1, self.config.num_iterations + 1):
            print(f"\n=== Iteration {iteration}/{self.config.num_iterations} ===")

            parent_idx = self.select_candidate()
            parent = self.prompt_pool[parent_idx]
            print(f"Selected parent (idx={parent_idx}, gen={parent.generation}, avg_score={parent.avg_score:.2f})")

            samples = self.generate_and_score_batch(
                parent.system_prompt, user_prompt, self.config.minibatch_size
            )
            parent_minibatch_score = sum(s[2] for s in samples) / len(samples)

            new_prompt = self.reflect_and_propose(
                parent.system_prompt, samples, user_prompt
            )
            print(f"Proposed new prompt: {new_prompt[:100]}...")

            new_samples = self.generate_and_score_batch(
                new_prompt, user_prompt, self.config.minibatch_size
            )
            new_minibatch_score = sum(s[2] for s in new_samples) / len(new_samples)

            improvement = new_minibatch_score - parent_minibatch_score
            print(f"Parent minibatch score: {parent_minibatch_score:.2f}")
            print(f"New prompt minibatch score: {new_minibatch_score:.2f}")
            print(f"Improvement: {improvement:.2f}")

            if improvement > self.config.improvement_threshold:
                full_samples = self.generate_and_score_batch(
                    new_prompt, user_prompt, self.config.samples_per_iteration
                )
                new_candidate = PromptCandidate(
                    system_prompt=new_prompt,
                    scores=[s[2] for s in full_samples],
                    generation=parent.generation + 1,
                    parent_idx=parent_idx,
                )
                self.prompt_pool.append(new_candidate)
                print(f"Added new candidate to pool (avg_score={new_candidate.avg_score:.2f})")

                self.prune_pool()
            else:
                print("No improvement, discarding new prompt")

            best_candidate = max(self.prompt_pool, key=lambda c: c.avg_score)
            wandb.log(
                {
                    "iteration": iteration,
                    "best_avg_score": best_candidate.avg_score,
                    "pool_size": len(self.prompt_pool),
                    "generation": best_candidate.generation,
                    "parent_minibatch_score": parent_minibatch_score,
                    "new_minibatch_score": new_minibatch_score,
                    "improvement": improvement,
                }
            )

        best_candidate = max(self.prompt_pool, key=lambda c: c.avg_score)
        print(f"\n=== Training Complete ===")
        print(f"Best prompt (gen={best_candidate.generation}, avg_score={best_candidate.avg_score:.2f}):")
        print(best_candidate.system_prompt)

        return best_candidate


def main():
    wandb.init(
        project="vequinox",
        config={
            "method": "gepa",
            "base_model": MODEL_NAME,
            "reflection_model": REFLECTION_MODEL,
        },
    )

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    config = GEPAConfig(
        num_iterations=10,
        samples_per_iteration=4,
        pool_size=5,
        minibatch_size=2,
        improvement_threshold=0.5,
    )

    trainer = GEPATrainer(model, tokenizer, config)

    best_candidate = trainer.train(USER_PROMPT)

    wandb.log(
        {
            "final_best_prompt": best_candidate.system_prompt,
            "final_best_score": best_candidate.avg_score,
            "final_generation": best_candidate.generation,
        }
    )

    wandb.finish()


if __name__ == "__main__":
    main()
