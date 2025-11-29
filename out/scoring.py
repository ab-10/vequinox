from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from io import BytesIO, StringIO
# import cairosvg

# def load_svg_from_string(svg_text, width=512, height=512):
#     """Convert SVG string to PIL Image"""

#     png_data = cairosvg.svg2png(
#         bytestring=svg_text.encode("utf-8"), output_width=width, output_height=height
#     )
#     return Image.open(io.BytesIO(png_data))


def svg_text_to_pil(svg_text, width=512, height=512):
    drawing = svg2rlg(StringIO(svg_text))
    if drawing is None:
        return None
    # Scale to desired dimensions
    scale_x = width / drawing.width
    scale_y = height / drawing.height
    drawing.width = width
    drawing.height = height
    drawing.scale(scale_x, scale_y)

    bio = BytesIO()
    renderPM.drawToFile(drawing, bio, fmt="PNG")
    bio.seek(0)
    return Image.open(bio)


def load_clip():
    """Load CLIP model and processor"""
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    return processor, model


def score_images(
    images: list[Image.Image], prompt: str, clip_processor, clip_model
) -> torch.Tensor:
    """Get CLIP scores for how well images match prompt.

    Args:
        images: List of PIL Images to score
        prompt: Text prompt to compare against
        clip_processor: CLIP processor
        clip_model: CLIP model

    Returns:
        Tensor of scores
    """
    clip_inputs = clip_processor(text=[prompt], images=images, return_tensors="pt")
    clip_inputs = {k: v.to(clip_model.device) for k, v in clip_inputs.items()}

    with torch.no_grad():
        clip_outputs = clip_model(**clip_inputs)

    return clip_outputs.logits_per_image.squeeze()
