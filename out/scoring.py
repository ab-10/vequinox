import xml
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel
import cairosvg
import wandb

def load_svg_from_string(svg_text, width=512, height=512):
    """Convert SVG string to PIL Image"""

    png_data = cairosvg.svg2png(
        bytestring=svg_text.encode("utf-8"), output_width=width, output_height=height
    )
    return Image.open(io.BytesIO(png_data))


def load_clip():
    """Load CLIP model and processor"""
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    return processor, model


def score_svg(svg_texts: list[str], prompt: str, clip_processor, clip_model) -> dict:
    """Get CLIP scores for how well images match prompt.

    Args:
        svg_texts: List of SVG strings to score
        prompt: Text prompt to compare against
        clip_processor: CLIP processor
        clip_model: CLIP model

    Returns:
        Dict with 'images' (list of PIL Images), 'scores' (tensor of floats), and 'indices' (list of original indices)
    """
    width = clip_processor.image_processor.crop_size["width"]
    height = clip_processor.image_processor.crop_size["height"]

    images = []
    indices = []
    for idx, svg in enumerate(svg_texts):
        try:
            images.append(load_svg_from_string(svg, width=width, height=height))
            indices.append(idx)
        except xml.etree.ElementTree.ParseError:
            print(f"could not parse svg:\n{svg}")

    if not images:
        return {"images": [], "scores": torch.tensor([]), "indices": []}

    clip_inputs = clip_processor(text=[prompt], images=images, return_tensors="pt")
    clip_inputs = {k: v.to(clip_model.device) for k, v in clip_inputs.items()}

    with torch.no_grad():
        clip_outputs = clip_model(**clip_inputs)

    scores = clip_outputs.logits_per_image.squeeze()
    if scores.dim() == 0:
        score_list = [float(scores.item())]
    else:
        score_list = [float(s) for s in scores]

    wandb.log(
        {
            "svg_images": [
                wandb.Image(image, caption=f"score={score_list[i]:.4f}")
                for i, image in enumerate(images)
            ],
            "svg_scores": score_list,
        }
    )

    return {"images": images, "scores": scores, "indices": indices}
