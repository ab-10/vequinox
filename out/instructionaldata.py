"""
Instructional Data Generation Framework

This module provides a framework for generating instruction-output pairs using Claude.
It includes functions to send prompts to Claude, parse the resulting JSON pairs,
and save them to a CSV file for instruction tuning purposes.
"""

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anthropic


@dataclass
class InstructionOutputPair:
    """Represents a single instruction-output pair for instruction tuning."""

    instruction: str
    output: str

    def to_dict(self) -> dict[str, str]:
        """Convert the pair to a dictionary."""
        return {"instruction": self.instruction, "output": self.output}


class InstructionalDataGenerator:
    """
    A framework for generating instruction-output pairs using Claude.

    This class provides methods to send prompts to Claude, parse the resulting
    instruction-output pairs, and save them to a CSV file.
    """

    DEFAULT_SYSTEM_PROMPT = """You are an expert at creating high-quality instruction-output pairs for training language models.
When given a topic, constraint, or context, generate diverse and informative instruction-output pairs.
Each pair should have a clear instruction and a corresponding high-quality output.

Format your response as a JSON array of objects, where each object has "instruction" and "output" keys.
Example format:
[
    {"instruction": "Explain photosynthesis in simple terms.", "output": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen..."},
    {"instruction": "What are the main steps of photosynthesis?", "output": "The main steps are: 1) Light absorption by chlorophyll, 2) Water splitting to release oxygen..."}
]

Ensure each instruction is clear, specific, and actionable.
Ensure each output is accurate, comprehensive, and well-formatted."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """
        Initialize the InstructionalDataGenerator.

        Args:
            model: The Claude model to use for generation.
            api_key: The Anthropic API key. If not provided, will use ANTHROPIC_API_KEY env var.
            system_prompt: Custom system prompt for Claude. If not provided, uses default.
        """
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def generate_pairs(
        self,
        prompt: str,
        num_pairs: int = 5,
        constraint: Optional[str] = None,
        context: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> list[InstructionOutputPair]:
        """
        Generate instruction-output pairs using Claude.

        Args:
            prompt: The main topic or instruction for generating pairs.
            num_pairs: The number of instruction-output pairs to generate.
            constraint: Optional constraint to apply to the generation.
            context: Optional context to provide for the generation.
            max_tokens: Maximum tokens for Claude's response.

        Returns:
            A list of InstructionOutputPair objects.

        Raises:
            anthropic.APIError: If there's an error communicating with the API.
            json.JSONDecodeError: If the response cannot be parsed as JSON.
        """
        user_message = self._build_user_message(prompt, num_pairs, constraint, context)

        response = self._send_to_claude(user_message, max_tokens)

        return self._parse_response(response)

    def _build_user_message(
        self,
        prompt: str,
        num_pairs: int,
        constraint: Optional[str],
        context: Optional[str],
    ) -> str:
        """
        Build the user message to send to Claude.

        Args:
            prompt: The main topic or instruction.
            num_pairs: Number of pairs to generate.
            constraint: Optional constraint.
            context: Optional context.

        Returns:
            The formatted user message.
        """
        message_parts = [f"Generate {num_pairs} instruction-output pairs about: {prompt}"]

        if constraint:
            message_parts.append(f"\nConstraint: {constraint}")

        if context:
            message_parts.append(f"\nContext: {context}")

        message_parts.append(
            "\n\nRespond with ONLY a valid JSON array of objects with 'instruction' and 'output' keys."
        )

        return "".join(message_parts)

    def _send_to_claude(self, user_message: str, max_tokens: int) -> str:
        """
        Send a message to Claude and get the response.

        Args:
            user_message: The message to send.
            max_tokens: Maximum tokens for the response.

        Returns:
            The text content of Claude's response.

        Raises:
            anthropic.APIError: If there's an error communicating with the API.
        """
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        return message.content[0].text

    def _parse_response(self, response: str) -> list[InstructionOutputPair]:
        """
        Parse Claude's response into InstructionOutputPair objects.

        Args:
            response: The raw response text from Claude.

        Returns:
            A list of InstructionOutputPair objects.

        Raises:
            json.JSONDecodeError: If the response cannot be parsed as JSON.
            ValueError: If the response format is invalid.
        """
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        data = json.loads(response)

        if not isinstance(data, list):
            raise ValueError("Response must be a JSON array")

        pairs = []
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Each item must be a JSON object")
            if "instruction" not in item or "output" not in item:
                raise ValueError("Each item must have 'instruction' and 'output' keys")

            pairs.append(
                InstructionOutputPair(
                    instruction=str(item["instruction"]),
                    output=str(item["output"]),
                )
            )

        return pairs


def save_pairs_to_csv(
    pairs: list[InstructionOutputPair],
    filepath: str | Path = "instructiontuning.csv",
    append: bool = False,
) -> Path:
    """
    Save instruction-output pairs to a CSV file.

    Args:
        pairs: List of InstructionOutputPair objects to save.
        filepath: Path to the CSV file. Defaults to "instructiontuning.csv".
        append: If True, append to existing file. If False, overwrite.

    Returns:
        The Path object of the saved file.
    """
    filepath = Path(filepath)
    mode = "a" if append and filepath.exists() else "w"
    write_header = not (append and filepath.exists())

    with open(filepath, mode, newline="", encoding="utf-8") as csvfile:
        fieldnames = ["instruction", "output"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        for pair in pairs:
            writer.writerow(pair.to_dict())

    return filepath


def load_pairs_from_csv(filepath: str | Path) -> list[InstructionOutputPair]:
    """
    Load instruction-output pairs from a CSV file.

    Args:
        filepath: Path to the CSV file to load.

    Returns:
        A list of InstructionOutputPair objects.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    pairs = []
    with open(filepath, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pairs.append(
                InstructionOutputPair(
                    instruction=row["instruction"],
                    output=row["output"],
                )
            )

    return pairs


def main() -> None:
    """
    Example usage of the InstructionalDataGenerator.

    This function demonstrates how to use the framework to generate
    instruction-output pairs and save them to a CSV file.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Warning: ANTHROPIC_API_KEY not set. Please set it to use this module.")
        print("Example usage:")
        print("  export ANTHROPIC_API_KEY='your-api-key'")
        print("  python instructionaldata.py")
        return

    generator = InstructionalDataGenerator()

    pairs = generator.generate_pairs(
        prompt="Python programming best practices",
        num_pairs=5,
        constraint="Focus on code readability and maintainability",
        context="For intermediate Python developers",
    )

    print(f"Generated {len(pairs)} instruction-output pairs:")
    for i, pair in enumerate(pairs, 1):
        print(f"\n--- Pair {i} ---")
        print(f"Instruction: {pair.instruction}")
        print(f"Output: {pair.output[:100]}..." if len(pair.output) > 100 else f"Output: {pair.output}")

    filepath = save_pairs_to_csv(pairs)
    print(f"\nSaved pairs to: {filepath}")


if __name__ == "__main__":
    main()
