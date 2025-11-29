import random
# import anthropic
from trl import BasePairwiseJudge



class PairwiseJudge(BasePairwiseJudge):
    def __init__(self, model="claude-sonnet-4-20250514"):
        self.model = model
        # self.client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

    def judge(self, prompts, completions, shuffle_order=True):
        results = []
        for prompt, (a, b) in zip(prompts, completions):
            swapped = shuffle_order and random.random() < 0.5
            if swapped:
                a, b = b, a

            # OVERRIDE: hardcoded judge
            # response = self.client.messages.create(
            #     model=self.model,
            #     max_tokens=1,
            #     messages=[{
            #         "role": "user",
            #         "content": f"Which response is better? Reply only A or B.\n\nPrompt: {prompt}\n\nA: {a}\n\nB: {b}"
            #     }]
            # )
            # choice = response.content[0].text.strip().upper()
            choice = "A"

            if choice == "A":
                results.append(1 if swapped else 0)
            elif choice == "B":
                results.append(0 if swapped else 1)
            else:
                results.append(-1)
        return results
