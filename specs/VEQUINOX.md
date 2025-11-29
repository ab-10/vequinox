# vequinox

Goal: post-train an LLM to generate SVGs using RL feedback from a larger model

First iteration: train it to generate images of pelicans.

Base deliverable: train it to generate images of pelicans riding bicycles.

End target: train it to generate images based on free text prompts.

## Setup

Base model: a small ~1B param LLM.

TODO: maybe we can use an even smaller LLM to quickly iterate before a larger run.

Guide model: Claude Haiku 4.5


## Scorer

Approach 1: We use a text-to-image matching model that can score how well an SVG matches a text prompt.

- CLIP Score - Measures how well an image matches a text prompt. You'd compute the CLIP score for each image against your prompt, then compare. Higher score = better match.
- ImageReward - Specifically trained on human preferences for text-to-image generation. It directly predicts which images humans would prefer.
- PickScore - Another human preference model trained on the Pick-a-Pic dataset of human comparisons.
- HPSv2 (Human Preference Score) - Trained to predict human aesthetic preferences.

As a baseline we can use CLIP score for initial testing. We can later integrate more sophisticated scoring models like ImageReward or PickScore for better feedback based on human preferences.


## Training Paradigm

1. Prompt the base model to generate multiple
2. Have a base model generate multiple 
