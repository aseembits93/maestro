import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

from maestro.trainer.common.utils.device import parse_device_spec


def predict_with_inputs(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    device: torch.device,
    max_new_tokens: int = 1024,
) -> list[str]:
    """Generate text predictions from preprocessed model inputs."""
    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = attention_mask.to(device, non_blocking=True)
    pixel_values = pixel_values.to(device, non_blocking=True)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
        )
        prefix_length = input_ids.shape[-1]
        generated_ids = generated_ids[:, prefix_length:]

    return processor.batch_decode(generated_ids, skip_special_tokens=True)


def predict(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    image: str | bytes | Image.Image,
    prefix: str,
    device: str | torch.device = "auto",
    max_new_tokens: int = 1024,
) -> str:
    """Generate a text prediction for a single image and text prefix."""
    device = parse_device_spec(device)
    text = "<image>" + prefix
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

    return predict_with_inputs(
        **inputs, model=model, processor=processor, device=device, max_new_tokens=max_new_tokens
    )[0]
