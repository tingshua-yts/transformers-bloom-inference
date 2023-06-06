deepspeed --num_gpus 2 bloom-inference-scripts/bloom-ds-zero-inference.py \
          --name /mnt/model/bloom-7b1 --batch_size 2 \
          --cpu_offload

