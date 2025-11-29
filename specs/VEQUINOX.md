---
title: vequinox
emoji: 🦅
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.8.0"
app_file: out/app.py
pinned: false
---

# vequinox

Goal: post-train an LLM to generate SVGs using RL feedback from a larger model

First iteration: train it to generate images of pelicans.

Base deliverable: train it to generate images of pelicans riding bicycles.

End target: train it to generate images based on free text prompts.

## Stack

1. Huggingface:
    1. Transfomers
    2. Datasets
    3. TRL
2. UV: managing the python environment

## Setup

Base model: `Qwen/Qwen3-0.6B`

Guide model: Claude Haiku 4.5


## Scorer

Approach 1: We use a text-to-image matching model that can score how well an SVG matches a text prompt.

- CLIP Score - Measures how well an image matches a text prompt. You'd compute the CLIP score for each image against your prompt, then compare. Higher score = better match.
- ImageReward - Specifically trained on human preferences for text-to-image generation. It directly predicts which images humans would prefer.
- PickScore - Another human preference model trained on the Pick-a-Pic dataset of human comparisons.
- HPSv2 (Human Preference Score) - Trained to predict human aesthetic preferences.

As a baseline we can use CLIP score for initial testing. We can later integrate more sophisticated scoring models like ImageReward or PickScore for better feedback based on human preferences.


## Training Paradigm

We use Online DPO via TRL's `OnlineDPOTrainer` with a custom pairwise judge backed by Claude.

1. **Candidate generation**
   - For each prompt in the batch, the base model generates 2 SVG candidates.
2. **Pairwise scoring**
   - The guide model (Claude Haiku 4.5) receives one pair at a time: the prompt text and both SVG candidates (A and B).
   - Claude evaluates using the rubric (semantic match, SVG validity, simplicity/clean shapes, safety) and returns which candidate is better.
3. **Online DPO update**
   - `OnlineDPOTrainer` uses the pairwise judgments to compute the DPO loss and update the policy in a single step.
   - No separate preference dataset accumulation—training happens online as candidates are generated and judged.
4. **Iterative refinement**
   - The loop continues: generate pairs → judge → update weights.
   - Monitor guide model win rates, SVG diversity, and qualitative checks.
   - Once pelican-bicycle performance is satisfactory, expand the prompt distribution toward free-text SVG generation.
