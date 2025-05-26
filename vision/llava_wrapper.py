import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, low_cpu_mem_usage=True
).to("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

SYSTEM_PROMPT = (
    "You are an AI visual assistant. "
    "You will be given an image and a related question. "
    "Answer the question in a concise and informative way based only on what you see in the image."
)

def generate_answer(image: np.ndarray, question: str) -> str:
    pil_image = Image.fromarray(image)

    # Make image 500x500 if it's larger
    if pil_image.size[0] > 500 or pil_image.size[1] > 500:
        pil_image = pil_image.resize((500, 500))

    full_prompt = f"{SYSTEM_PROMPT}\n<image>\nQuestion: {question}, Answer:"

    inputs = processor(text=full_prompt, images=pil_image, return_tensors="pt").to(model.device)

    # Generate the answer
    output = model.generate(**inputs, max_new_tokens=200)
    decoded = processor.batch_decode(output, skip_special_tokens=True)[0]

    # Delete potential "Answer:" prefix
    if "Answer:" in decoded:
        decoded = decoded.split("Answer:")[-1].strip()

    return decoded
