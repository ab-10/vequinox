PREFIX = """
Assistant:
```xml
<svg width="200" height="120" xmlns="http://www.w3.org/2000/svg">
"""

MODEL_NAME = "Qwen/Qwen3-0.6B"
SYS_PROMPT = "You are an SVG generator. Respond only with valid SVG code. /no_think"
PROMPT = [
    {"role": "system", "content": SYS_PROMPT},
    {"role": "user", "content": "Generate SVG code of a pelican riding a bicycle"},
]
