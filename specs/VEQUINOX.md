# vequinox

Goal: post-train an LLM to generate SVGs using RL feedback from a larger model

First iteration: train it to generate images of pelicans.

Base deliverable: train it to generate images of pelicans riding bicycles.

End target: train it to generate images based on free text prompts.

## Setup

Base model: a small ~1B param LLM.

TODO: maybe we can use an even smaller LLM to quickly iterate before a larger run.

Guide model: Claude Haiku 4.5

## Training Paradigm

1. Prompt the base model to generate multiple
2. Have a base model generate multiple 
