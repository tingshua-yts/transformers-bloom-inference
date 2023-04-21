from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
import torch
import os
import logging
import torch
logging.basicConfig(format='[%(asctime)s] %(filename)s %(funcName)s():%(lineno)i [%(levelname)s] %(message)s', level=logging.DEBUG)

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

logging.info(f"rank[{local_rank}]: start rank ")
model_name="/mnt/model/bloom-7b1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
logging.info(f"rank[{local_rank}]:after load tokenizer")

with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
    model = AutoModelForCausalLM.from_pretrained(model_name)

logging.info(f"rank[{local_rank}]: after load model")

model = model.eval()
model = deepspeed.init_inference(
    model,
    mp_size=world_size,
    base_dir="/mnt/model/bloom-7b1",
    dtype=torch.float16,
    checkpoint="/mnt/project/transformers-bloom-inference/example/bloom-7b/ds_checkpoint.json",
    #replace_method='auto',
    replace_with_kernel_inject=True
)

# tokenize
encoded_inputs = tokenizer("deepspeed is", padding=True, return_tensors='pt')

# forward
model_outputs = model.generate(input_ids=encoded_inputs["input_ids"].to(f"cuda:{local_rank}"), attention_mask=encoded_inputs["attention_mask"])

# detokenize
text = tokenizer.decode( model_outputs[0],skip_special_tokens=True,  )
print(f"naive result: {text}")