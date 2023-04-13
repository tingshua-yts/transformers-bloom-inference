deepspeed --num_gpus 2 --module inference_server.benchmark \
          --model_name bigscience/bloom-560m \
          --model_class AutoModelForCausalLM --dtype fp16 \
          --deployment_framework ds_inference --benchmark_cycles 2