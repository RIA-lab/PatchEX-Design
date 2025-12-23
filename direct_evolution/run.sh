OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python main.py --task GB1 --oracle wetlab --patience 3 --max_oracle_call_per_step 100 --internal_steps 100 --eta 0.1 --parallel_samples 128
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python main.py --task PhoQ --oracle wetlab --patience 3 --max_oracle_call_per_step 100 --internal_steps 100 --eta 0.1 --parallel_samples 128
