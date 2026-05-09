import torch
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained('allenai/Molmo-7B-O-0924', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-O-0924',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # or torch.float16
    device_map='auto',
    load_in_4bit=True,  # Quantizes to 4-bit, fits most GPUs
    low_cpu_mem_usage=True
)