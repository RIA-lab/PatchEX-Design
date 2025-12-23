import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
from .sample_model import SampleModel
from .model.ensemble import Ensemble
from .oracle.wetlab import WetLab_Landscape
from .utils.seq_utils import hamming_distance, mutation_alphabet, random_mutation, sequence_to_one_hot
from .model.psi import PSI

logger = logging.getLogger('directed_evolution')

def run(config_de):
    selected_residues = ''.join([config_de.wt_seq[_] for _ in config_de.selected_residue_idx])

    logger.info(config_de)

    psi_model_ensemble = Ensemble([PSI(config_de, len(mutation_alphabet), config_de.embed_dim, config_de.hidden_dim, len(selected_residues)).to(config_de.device) for i in range(config_de.ensemble)], 'ucb')
    
    data_store = dict({'step':[], 'sequence': [], 'fitness': [], 'distance': []})

    if config_de.task == 'ph':
        sigma = 0.3
    elif config_de.task == 'opt':
        sigma = 4.0
    else:
        raise ValueError

    oracle = WetLab_Landscape(config_de.oracle_model, config_de.wt_accession, config_de.selected_residue_idx, config_de.wt_seq, config_de.target_value, sigma=sigma)

    wt_fitness = oracle.get_wt_fitness()
    data_store['step'].append(0)
    data_store['sequence'].append(selected_residues)
    data_store['fitness'].append(wt_fitness)
    data_store['distance'].append(0)
    logger.info(f'wt fitness: {wt_fitness}')

    sample_model = SampleModel(wt_fitness, config_de, psi_model_ensemble)

    seq_scores = dict({selected_residues: wt_fitness})
    max_fitness = wt_fitness
    max_seq = selected_residues
    seed_seqs = [selected_residues]
    all_seqs = set([selected_residues])


    for i in range(1, config_de.evo_steps + 1):
        logger.info("===EVO STEP {}===".format(i))
        
        samples = set(seed_seqs)
        # random samples for first round
        while i == 1 and len(samples) < config_de.max_oracle_call_per_step:
            for seq in seed_seqs:
                mutant = random_mutation(seq, mutation_alphabet, -1)
                if mutant not in all_seqs and oracle.is_valid_seq(mutant):
                    samples.add(mutant)
                    if len(samples) >= config_de.max_oracle_call_per_step:
                        break
        
        samples = list(samples)[:config_de.max_oracle_call_per_step]

        samples_fitness = []
        samples_seq_scores = dict()
        prev_max_fitness = max_fitness
        seed_seqs = [max_seq]
        for bi in tqdm(range(0, len(samples), config_de.oracle_batch_size)):
            sample_batch = samples[bi:bi + config_de.oracle_batch_size]
            fitness = oracle.get_fitness(sample_batch)  # bsz x 1
            samples_fitness += fitness

            batch_max = np.max(fitness)
            if  batch_max > max_fitness:
                max_fitness = batch_max
                max_seq = sample_batch[np.argmax(fitness)]

            for si in range(len(sample_batch)):
                seq_scores[sample_batch[si]] = fitness[si]
                samples_seq_scores[sample_batch[si]] = fitness[si]
                if fitness[si] > prev_max_fitness:
                    seed_seqs.append(sample_batch[si])

        all_seqs.update(samples)
        logger.info('{} sampled sequences fitness: max: {:.4f}, min: {:.4f}, avg: {:.4f}'.format(len(samples_fitness), np.max(samples_fitness), np.min(samples_fitness), np.mean(samples_fitness)))

        logger.info('global measured sequences count: {}'.format(len(all_seqs)))
        logger.info('global max fitness: {:.4f}, distance to wt: {}'.format(max_fitness, hamming_distance(selected_residues, max_seq)))

        data_store['step'] += [i for j in range(len(samples))]
        data_store['sequence'] += samples
        data_store['fitness'] += samples_fitness
        data_store['distance'] += [hamming_distance(selected_residues, seq) for seq in samples]

        if i == config_de.evo_steps:
            # save model parameters
            for mi, psi_model in enumerate(psi_model_ensemble.models):
                torch.save(psi_model.state_dict(), f'{config_de.de_out}/surrogate_model_{mi}.ckpt')
            return data_store

        all_seq_scores = list(seq_scores.items())
        sample_seqs_scores = list(samples_seq_scores.items())


        # training fitness surrogate
        for mi, psi_model in enumerate(psi_model_ensemble.models):
            optimizer = optim.Adam(params=psi_model.parameters(), lr=config_de.lr)
            loss_fn = nn.MSELoss()
            psi_model.train()
            psi_model.struct_encoder.eval()
            
            logger.info(f"fine-tuning fitness model {mi}:")
            epoch_losses = []
            patience = 0
            min_loss = torch.inf
            epoch = 1
            while patience < config_de.patience:
                losses = []
                random.shuffle(all_seq_scores)
                for bi in range(0, len(all_seq_scores), config_de.batch_size):
                    optimizer.zero_grad()
                    batch_data = all_seq_scores[bi:bi + config_de.batch_size]
                    labels = torch.tensor([data[1] for data in batch_data], dtype=torch.float32).to(config_de.device)
                    input_onehot = torch.stack([sequence_to_one_hot(data[0], mutation_alphabet) for data in batch_data], dim=0).float().to(config_de.device)

                    with torch.autograd.set_detect_anomaly(True):
                        output = psi_model.forward(input_onehot, False)
                        loss = loss_fn(output, labels)
                        losses.append(loss.item())
                        loss.backward()
                        optimizer.step()
                
                epoch_loss = np.mean(losses)
                epoch_loss_std = np.std(losses)
                epoch_losses.append(epoch_loss)
                if epoch == 1:
                    logger.info(f'epoch {epoch}\tloss: {epoch_losses[-1]:.4f}, std: {epoch_loss_std:.4f}')
                epoch += 1
                if epoch_loss < min_loss:
                    min_loss = epoch_loss
                    patience = 0
                else:
                    patience += 1
                
                if patience >= config_de.patience:
                    logger.info(f'epoch {epoch}\tloss: {epoch_losses[-1]:.4f}, std: {epoch_loss_std:.4f}')
            
            psi_model.eval()
        
        # test psi model performance on collected samples with surrogate
        pred_scores = [[], [], []]
        for mi, psi_model in enumerate(psi_model_ensemble.models):
            psi_model.eval()
            for bi in range(0, len(all_seq_scores), config_de.batch_size):
                batch_data = all_seq_scores[bi:bi + config_de.batch_size]
                labels = torch.tensor([data[1] for data in batch_data], dtype=torch.float32).to(config_de.device)
                input_onehot = torch.stack([sequence_to_one_hot(data[0], mutation_alphabet) for data in batch_data], dim=0).float().to(config_de.device)
                output = psi_model.forward(input_onehot)
                pred_scores[mi] += output.tolist()
        
        aes = []
        for i in range(len(all_seq_scores)):
            aes.append(sum([abs(pred_scores[m][i] - all_seq_scores[i][1]) for m in range(config_de.ensemble)]) / config_de.ensemble)
        logger.info(f'uncertainty: {np.mean(aes)}')

        seed_seqs = [max_seq]
        if config_de.sampler == 'lmc':
            logger.info(f'lmc sampling:')
            esm_samples = sample_model.lmc_psi_sample(seed_seqs, all_seqs)
        elif config_de.sampler == 'hmc':
            logger.info(f'hmc sampling:')
            esm_samples = sample_model.hmc_psi_sample(seed_seqs, all_seqs)
        elif config_de.sampler == 'random':
            logger.info(f'random sampling:')
            esm_samples = sample_model.random_sample(seed_seqs, all_seqs)
        else:
            assert config_de.sampler in ['lmc', 'hmc', 'random']
        
        seed_seqs = list(esm_samples)

    return data_store
