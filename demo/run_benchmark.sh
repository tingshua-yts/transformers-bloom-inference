deepspeed --num_gpus 8 --module inference_server.benchmark \
          --model_name microsoft/bloom-deepspeed-inference-fp16 \
          --model_class AutoModelForCausalLM --dtype fp16 \
          --deployment_framework ds_inference --benchmark_cycles 5
