

# without Hamiltonian
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -u main.py --task GB1 --sampler random --use_structure 1 --max_oracle_call_per_step 100 --evo_steps 10 --internal_steps 16 --seed 2024
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python -u main.py --task PhoQ --sampler random --use_structure 1 --max_oracle_call_per_step 100 --evo_steps 10 --internal_steps 16 --seed 2024

# without virutal barrier
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python -u main.py --task GB1 --sampler hmc --onehot_constraint 0 --use_structure 1 --max_oracle_call_per_step 100 --evo_steps 10 --internal_steps 16 --seed 2024
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 python -u main.py --task PhoQ --sampler hmc --onehot_constraint 0 --use_structure 1 --max_oracle_call_per_step 100 --evo_steps 10 --internal_steps 16 --seed 2024

# without std
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -u main.py --task GB1 --sampler hmc --ensemble 1 --use_structure 1 --max_oracle_call_per_step 100 --evo_steps 10 --internal_steps 16 --seed 2024
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python -u main.py --task PhoQ --sampler hmc --ensemble 1 --use_structure 1 --max_oracle_call_per_step 100 --evo_steps 10 --internal_steps 16 --seed 2024

# without structure
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python -u main.py --task GB1 --sampler hmc --use_structure 0 --max_oracle_call_per_step 100 --evo_steps 10 --internal_steps 16 --seed 2024
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 python -u main.py --task PhoQ --sampler hmc --use_structure 0 --max_oracle_call_per_step 100 --evo_steps 10 --internal_steps 16 --seed 2024
