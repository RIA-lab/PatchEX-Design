import torch
import torch.nn.functional as F
import numpy as np
import logging

from .model.ensemble import Ensemble
from .utils.seq_utils import random_mutation, index_to_sequence, sequence_to_one_hot, hamming_distance, mutation_alphabet

logger = logging.getLogger('sample_model')

class SampleModel:
    def __init__(self, wt_fitness, args, psi_model_ensemble: Ensemble):
        self.args = args
        self.device = args.device
        self.psi_model_ensemble = psi_model_ensemble
        self.wt_fitness = wt_fitness
    
    # algorithm 2 in paper
    def hmc_psi_sample(self, seeds, all_seqs, valid_seqs=None):
        logger.info('hmc sampling...')
        seeds_onehot = torch.stack([sequence_to_one_hot(seed, mutation_alphabet) for seed in seeds], dim=0).to(self.device)
        seeds_onehot = seeds_onehot.repeat(self.args.parallel_samples // len(seeds) + 1, 1, 1)[:self.args.parallel_samples].float()

        curr_samples = seeds_onehot.clone()
        seq_len = curr_samples.size(1)

        x_rank = len(curr_samples.shape) - 1

        pos_mask = torch.ones_like(curr_samples).to(curr_samples.device)
        
        pos_mask[:,0:seq_len] = 0
        
        pos_mask = pos_mask.bool()
        pos_mask = pos_mask.reshape(self.args.parallel_samples,-1)

        collected_seqs = set()
        all_seq_scores = dict()
        for mi, psi_model in enumerate(self.psi_model_ensemble.models):
            logger.info(f'sampling model {mi+1}')

            step = 1
            
            eps = self.args.eta
            loop = True
            seq_scores = []
            model_sampled_seqs = set()

            with torch.enable_grad():
                psi_model.zero_grad()
                U_seeds, _ = psi_model.get_energy_and_grad(seeds_onehot)
            while loop and step <= self.args.max_sample_steps:
                logger.info(f'loop step {step}, len(model_sampled_seqs): {len(model_sampled_seqs)}')

                q = seeds_onehot.clone()
                q_onehot = seeds_onehot.clone()
                
                traj_list = [q]
                p = torch.randn_like(q).to(self.args.device)
                K_p = torch.sum(p.pow(2), dim=[1,2])/2
                
                psi_log_ratio = []
                l = 1
                for i in range(self.args.internal_steps):
                    l += 1
                    with torch.enable_grad():
                        psi_model.zero_grad()
                        U_q, grad_U_q = psi_model.get_energy_and_grad(q)
                        grad_U_q = F.normalize(grad_U_q, p=2, dim=2)
                    
                    p_half = p - eps / 2 * grad_U_q     # formula 2
                    if self.args.onehot_constraint:
                        p_i = p_half.clone()
                        q_i = q + eps * p_i
                        valid = ((q_i >= 0) & (q_i <= 1)).sum().item()

                        # formula 6-7
                        while valid != q_i.size(0) * q_i.size(1) * q_i.size(2):
                            u_idx = torch.nonzero(q_i > 1)
                            q_i[u_idx[:,0], u_idx[:,1], u_idx[:,2]] = 2 - q_i[u_idx[:,0], u_idx[:,1], u_idx[:,2]] # formula 7
                            p_i[u_idx[:,0], u_idx[:,1], u_idx[:,2]] = -p_i[u_idx[:,0], u_idx[:,1], u_idx[:,2]] # formula 6
                            
                            l_index = torch.nonzero(q_i < 0).squeeze(-1)
                            q_i[l_index[:,0], l_index[:,1], l_index[:,2]] = -q_i[l_index[:,0], l_index[:,1], l_index[:,2]] # formula 7
                            p_i[l_index[:,0], l_index[:,1], l_index[:,2]] = -p_i[l_index[:,0], l_index[:,1], l_index[:,2]] # formula 6
                            
                            valid = ((q_i >= 0) & (q_i <= 1)).sum().item()
                        
                        new_q = q_i
                        p_half = p_i
                    else:
                        new_q = q + eps * p_half

                    with torch.enable_grad():
                        psi_model.zero_grad()
                        U_q, grad_U_new_q = psi_model.get_energy_and_grad(new_q)
                        grad_U_new_q = F.normalize(grad_U_new_q, p=2, dim=2)

                    new_p = p_half - eps / 2 * grad_U_new_q

                    p = new_p.clone()
                    q = new_q.clone()

                    # line 6 in algorithm 2
                    new_q_onehot = F.one_hot(torch.argmax(new_q, dim=-1), num_classes=new_q.size(-1)).to(self.args.device).float()
                    K_new_p = torch.sum(new_p.pow(2), dim=[1,2])/2
                    U_new_q, grad_U_new_q = psi_model.get_energy_and_grad(new_q_onehot)
                    
                    # line 7 in algorithm 2
                    accepted = (torch.exp(U_seeds - U_new_q + K_p - K_new_p) >= torch.rand_like(U_q)).float().view(-1, *([1] * x_rank))
                    accepted_q = new_q_onehot * accepted + q_onehot * (1 - accepted)

                    q_onehot = accepted_q.clone()

                    traj_list.append(accepted_q.clone())

                    psi_log_ratio = psi_model.forward(accepted_q)
                    proposed_seq_score = dict()
                    for si in range(self.args.parallel_samples):
                        seq = index_to_sequence(torch.argmax(accepted_q[si], dim=-1).tolist(), mutation_alphabet)  # bsz
                        proposed_seq_score[seq] = psi_log_ratio[si].item()

                    for seq, score in proposed_seq_score.items():
                        if seq not in all_seqs and seq not in model_sampled_seqs and (valid_seqs == None or seq in valid_seqs):
                            seq_scores.append([seq, score])
                            model_sampled_seqs.add(seq)
                            if len(model_sampled_seqs) >= 100:
                                loop = False

                step += 1
            logger.info(f'len(model_sampled_seqs): {len(model_sampled_seqs)}')
            model_seq_scores = [ss for ss in sorted(seq_scores, key=lambda x:x[1], reverse=True)[:self.args.max_oracle_call_per_step]]

            for seq_score in model_seq_scores:
                all_seq_scores[seq_score[0]] = seq_score[1] if seq_score[0] not in all_seq_scores else max(seq_score[1], all_seq_scores[seq_score[0]])
        
        sampled_seqs = list(all_seq_scores.keys())
        sampled_seq_bound = []

        for bi in range(0, len(sampled_seqs), self.args.batch_size):
            batch_input = torch.stack([sequence_to_one_hot(seq, mutation_alphabet) for seq in sampled_seqs[bi:bi+self.args.batch_size]]).float().to(self.args.device)
            batch_output = self.psi_model_ensemble.forward(batch_input, 'ucb')
            sampled_seq_bound += [[sampled_seqs[bi+si], batch_output[si].item()] for si in range(len(batch_output))]

        collected_seqs = [ss[0] for ss in sorted(sampled_seq_bound, key=lambda x:x[1], reverse=True)[:self.args.max_oracle_call_per_step]]

        logger.info('len(collected_seqs): {}'.format(len(collected_seqs)))
        distances = []
        for seq in collected_seqs:
            distances.append(hamming_distance(seq, seeds[0]))
        logger.info(f'collected distance: {np.mean(distances)}\t{np.std(distances)}')
        distances = []
        for s_b in sampled_seq_bound:
            distances.append(hamming_distance(s_b[0], seeds[0]))
        logger.info(f'sampled seqs bound distance: {np.mean(distances)}\t{np.std(distances)}')
        return collected_seqs

    def random_sample(self, seeds, all_seqs, valid_seqs=None):
        '''
            generate random mutations of seeds, evaluate top K samples
        '''
        all_seq_scores = dict()
        all_seqs_list = list(all_seqs)
        random_samples = set()
        while len(random_samples) < 2048:
            seed = np.random.choice(all_seqs_list)
            new_sample = random_mutation(seed, mutation_alphabet, -1)
            while new_sample in all_seqs:
                    new_sample = random_mutation(seed, mutation_alphabet, -1)
            random_samples.add(new_sample)
        
        model_samples = list(random_samples)
        for mi, psi_model in enumerate(self.psi_model_ensemble.models):
            model_seq_scores = []
            for bi in range(0, len(model_samples), self.args.batch_size):
                batch_seqs = model_samples[bi:bi+self.args.batch_size]
                batch_input = torch.stack([sequence_to_one_hot(seq, mutation_alphabet) for seq in batch_seqs]).float().to(self.args.device)
                batch_output = psi_model.forward(batch_input).tolist()
                model_seq_scores += [[batch_seqs[i], batch_output[i]] for i in range(len(batch_seqs))]

            for seq_score in model_seq_scores:
                all_seq_scores[seq_score[0]] = seq_score[1] if seq_score[0] not in all_seq_scores else max(seq_score[1], all_seq_scores[seq_score[0]])
        
        sampled_seqs = list(all_seq_scores.keys())
        sampled_seq_bound = []

        for bi in range(0, len(sampled_seqs), self.args.batch_size):
            batch_input = torch.stack([sequence_to_one_hot(seq, mutation_alphabet) for seq in sampled_seqs[bi:bi+self.args.batch_size]]).float().to(self.args.device)
            batch_output = self.psi_model_ensemble.forward(batch_input, 'ucb')
            sampled_seq_bound += [[sampled_seqs[bi+si], batch_output[si].item()] for si in range(len(batch_output))]

        collected_seqs = [ss[0] for ss in sorted(sampled_seq_bound, key=lambda x:x[1], reverse=True)[:self.args.max_oracle_call_per_step]]
        logger.info('len(collected_seqs): {}'.format(len(collected_seqs)))
        distances = []
        for seq in collected_seqs:
            distances.append(hamming_distance(seq, seeds[0]))
        logger.info(f'collected distance: {np.mean(distances)}\t{np.std(distances)}')
        distances = []
        for s_b in sampled_seq_bound:
            distances.append(hamming_distance(s_b[0], seeds[0]))
        logger.info(f'sampled seqs bound distance: {np.mean(distances)}\t{np.std(distances)}')
        return collected_seqs

    def lmc_psi_sample(self, seeds, all_seqs, valid_seqs=None):
        logger.info('lmc trajectory sampling...')
        seeds_onehot = torch.stack([sequence_to_one_hot(seed, mutation_alphabet) for seed in seeds], dim=0).to(self.device)
        seeds_onehot = seeds_onehot.repeat(self.args.parallel_samples // len(seeds) + 1, 1, 1)[:self.args.parallel_samples].float()

        curr_samples = seeds_onehot.clone()
        seq_len = curr_samples.size(1)
        x_rank = len(curr_samples.shape) - 1

        pos_mask = torch.ones_like(curr_samples).to(curr_samples.device)
        pos_mask[:,0:seq_len] = 0
        
        pos_mask = pos_mask.bool()
        pos_mask = pos_mask.reshape(self.args.parallel_samples,-1)

        collected_seqs = set()
        all_seq_scores = dict()
        for mi, psi_model in enumerate(self.psi_model_ensemble.models):
            logger.info(f'sampling model {mi+1}')

            step = 1
            
            eps = self.args.eta
            loop = True
            seq_scores = []
            model_sampled_seqs = set()

            with torch.enable_grad():
                psi_model.zero_grad()
                U_seeds, _ = psi_model.get_energy_and_grad(seeds_onehot)
            q = seeds_onehot.clone()
            q_onehot = seeds_onehot.clone()
            while loop and step <= self.args.max_sample_steps:
                logger.info(f'loop step {step}, len(model_sampled_seqs): {len(model_sampled_seqs)}')

                traj_list = [q]
                p = torch.randn_like(q).to(self.args.device)
                K_p = torch.sum(p.pow(2), dim=[1,2])/2

                psi_log_ratio = []
                with torch.enable_grad():
                    psi_model.zero_grad()
                    U_q, grad_U_q = psi_model.get_energy_and_grad(q)
                    grad_U_q = F.normalize(grad_U_q, p=2, dim=2)
                
                p_half = p - eps / 2 * grad_U_q
                p_i = p_half.clone()
                q_i = q + eps * p_i
                upper = psi_model.vocab_size - 1
                lower = 0
                valid = ((q_i >= lower) & (q_i <= upper)).sum().item()
                while valid != q_i.size(0) * q_i.size(1) * q_i.size(2):
                    u_idx = torch.nonzero(q_i > upper)
                    q_i[u_idx[:,0], u_idx[:,1], u_idx[:,2]] = 2 - q_i[u_idx[:,0], u_idx[:,1], u_idx[:,2]]
                    p_i[u_idx[:,0], u_idx[:,1], u_idx[:,2]] = -p_i[u_idx[:,0], u_idx[:,1], u_idx[:,2]]
                    
                    l_index = torch.nonzero(q_i < lower).squeeze(-1)
                    q_i[l_index[:,0], l_index[:,1], l_index[:,2]] = -q_i[l_index[:,0], l_index[:,1], l_index[:,2]]
                    p_i[l_index[:,0], l_index[:,1], l_index[:,2]] = -p_i[l_index[:,0], l_index[:,1], l_index[:,2]]
                    
                    valid = ((q_i >= lower) & (q_i <= upper)).sum().item()
                
                new_q = q_i
                p_half = p_i

                with torch.enable_grad():
                    psi_model.zero_grad()
                    U_q, grad_U_new_q = psi_model.get_energy_and_grad(new_q)
                    grad_U_new_q = F.normalize(grad_U_new_q, p=2, dim=2)

                new_p = p_half - eps / 2 * grad_U_new_q
                q = new_q.clone()

                new_q_onehot = F.one_hot(torch.argmax(new_q, dim=-1), num_classes=new_q.size(-1)).to(self.args.device).float()

                K_new_p = torch.sum(new_p.pow(2), dim=[1,2])/2
                U_new_q, grad_U_new_q = psi_model.get_energy_and_grad(new_q_onehot)
                
                accepted = (torch.exp(U_seeds - U_new_q + K_p - K_new_p) >= torch.rand_like(U_q)).float().view(-1, *([1] * x_rank))
                accepted_q = new_q_onehot * accepted + q_onehot * (1 - accepted)
                q_onehot = accepted_q
                traj_list.append(accepted_q.clone())

                psi_log_ratio = psi_model.forward(accepted_q)
                proposed_seq_score = dict()
                for si in range(self.args.parallel_samples):
                    seq = index_to_sequence(torch.argmax(accepted_q[si], dim=-1).tolist(), mutation_alphabet)  # bsz
                    proposed_seq_score[seq] = psi_log_ratio[si].item()

                for seq, score in proposed_seq_score.items():
                    if seq not in all_seqs and seq not in model_sampled_seqs and seq in valid_seqs:
                        seq_scores.append([seq, score])
                        model_sampled_seqs.add(seq)
                        if len(model_sampled_seqs) >= 100:
                            loop = False
                step += 1
            
            logger.info(f'len(model_sampled_seqs): {len(model_sampled_seqs)}')
            model_seq_scores = [ss for ss in sorted(seq_scores, key=lambda x:x[1], reverse=True)[:self.args.max_oracle_call_per_step]]

            for seq_score in model_seq_scores:
                all_seq_scores[seq_score[0]] = seq_score[1] if seq_score[0] not in all_seq_scores else max(seq_score[1], all_seq_scores[seq_score[0]])
        
        sampled_seqs = list(all_seq_scores.keys())
        sampled_seq_bound = []

        for bi in range(0, len(sampled_seqs), self.args.batch_size):
            batch_input = torch.stack([sequence_to_one_hot(seq, mutation_alphabet) for seq in sampled_seqs[bi:bi+self.args.batch_size]]).float().to(self.args.device)
            batch_output = self.psi_model_ensemble.forward(batch_input, 'ucb')

            sampled_seq_bound += [[sampled_seqs[bi+si], batch_output[si].item()] for si in range(len(batch_output))]

        collected_seqs = [ss[0] for ss in sorted(sampled_seq_bound, key=lambda x:x[1], reverse=True)[:self.args.max_oracle_call_per_step]]
        logger.info('len(collected_seqs): {}'.format(len(collected_seqs)))
        distances = []
        for seq in collected_seqs:
            distances.append(hamming_distance(seq, seeds[0]))
        logger.info(f'collected distance: {np.mean(distances)}\t{np.std(distances)}')
        distances = []
        for s_b in sampled_seq_bound:
            distances.append(hamming_distance(s_b[0], seeds[0]))
        logger.info(f'sampled seqs bound distance: {np.mean(distances)}\t{np.std(distances)}')
        return collected_seqs