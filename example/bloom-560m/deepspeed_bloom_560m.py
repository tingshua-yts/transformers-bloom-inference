from transformers import pipeline
import deepspeed
import torch
import os
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

generator = pipeline("text-generation", model="bigscience/bloom-560m", device=local_rank)

generator.model = deepspeed.init_inference(
    generator.model,
    mp_size=world_size,
    dtype=torch.float16,
    replace_method='auto',
    replace_with_kernel_inject=True
)
result = generator("deepspeed is ")
print(f"deepspeed result:{result}")