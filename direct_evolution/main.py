import torch
import numpy as np
import logging
import random
import argparse
import time
import os
import pandas as pd
import directed_evolution as de

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--esm", default="esm2_t33_650M_UR50D", type=str, choices=["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D"])
    parser.add_argument("--task", default="GB1", type=str, choices=["GB1", "PhoQ"])
    parser.add_argument("--evo_steps", default=10, type=int)
    parser.add_argument("--max_oracle_call_per_step", default=100, type=int)
    parser.add_argument("--max_mutation", default=1, type=int)
    parser.add_argument("--max_sample_steps", default=100, type=int)
    parser.add_argument("--internal_steps", default=64, type=int)
    parser.add_argument("--device", default="cuda", type=str, choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--oracle_batch_size", default=16, type=int)
    parser.add_argument("--parallel_samples", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--oracle", default="wetlab", type=str, choices=["wetlab"])
    parser.add_argument("--dropout", default=0.1)
    parser.add_argument("--embed_dim", default=256, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--top_k", default=100, type=int)
    parser.add_argument("--patience", default=3, type=int)
    parser.add_argument("--eta", default=0.1, type=float)
    parser.add_argument("--ensemble", default=3, type=int)
    parser.add_argument("--run_times", default=10, type=int)
    parser.add_argument("--use_structure", default=0, type=int)
    parser.add_argument('--sampler', default='hmc', type=str, choices=['lmc', 'hmc', 'random'])
    parser.add_argument('--exp_name', default='default', type=str)
    parser.add_argument('--onehot_constraint', default=1, type=int)
    args = parser.parse_args()

    setup_seed(args.seed)
    
    tm = time.localtime()

    log_dir = f"log/{tm.tm_year}/{tm.tm_mon}/{tm.tm_mday}"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{log_dir}/log-{tm.tm_hour}-{tm.tm_min}-{tm.tm_sec}-{args.exp_name}.txt"
    logging.basicConfig(filename=log_filename, format='%(asctime)s-%(levelname)s-%(filename)s-%(lineno)d-%(message)s', level=logging.INFO)
    logger = logging.getLogger('main')

    logger.info("START")
    try:
        os.makedirs(f'results/{args.task}', exist_ok=True)
        data_filename = f'results/HADES-{args.exp_name}_{args.task}_results_wetlab.xlsx'
        with pd.ExcelWriter(data_filename, engine='xlsxwriter') as writer:
            for i in range(1, args.run_times + 1):
                results = de.run(args)
                df = pd.DataFrame(results)
                df.to_excel(writer, sheet_name=f'{i}', index=False)
    except Exception as ex:
        logger.exception(ex)