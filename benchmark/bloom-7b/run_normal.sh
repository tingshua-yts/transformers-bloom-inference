set -x
set -e
deepspeed --num_gpus  8 --module inference_server.benchmark \
          --model_name /mnt/model/bloom-7b1 \
          --model_class AutoModelForCausalLM \
          --dtype fp16 --deployment_framework ds_inference --benchmark_cycles 10 \
          --batch_size="8,16,32,64"
