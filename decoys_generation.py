#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:59:38 2026

@author: joao
"""
import pandas as pd
import torch
import esm
import random
import math
import argparse
import matplotlib.pyplot as plt
from random import shuffle
from tqdm import tqdm

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()

def insert_mask(sequence, position, mask="<mask>"):
    """
    Replaces a character in a given position of a sequence with a mask.

    Parameters:
    - sequence (str or list): The sequence to replace the character in.
    - position (int): The position in the sequence where the character should be replaced.
    - mask (str): The mask to insert (default is "<mask>").

    Returns:
    - str or list: The sequence with the mask replacing the character at the specified position.
    """
    
    if not (0 <= position < len(sequence)):
        raise ValueError("Position is out of bounds.")
    
    if isinstance(sequence, str):
        return sequence[:position] + mask + sequence[position + 1:]
    elif isinstance(sequence, list):
        return sequence[:position] + [mask] + sequence[position + 1:]
    else:
        raise TypeError("Sequence must be a string or list.")

def sample_from_lowest_nonzero(probs, n=5):
    """
    Samples from the n lowest non-zero probabilities.
    
    Args:
        probs (torch.Tensor): Probability tensor of shape (1, vocab_size)
        n (int): Number of lowest non-zero candidates to consider
    
    Returns:
        int: Sampled token index
    """
    probs = probs.squeeze(0)  # (vocab_size,)
    
    # Mask zeros — only keep non-zero positions
    nonzero_mask = probs > 0
    nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0]  # indices where prob > 0
    nonzero_probs = probs[nonzero_indices]
    
    # Get the n lowest among non-zero
    k = min(n, len(nonzero_probs))
    lowest_vals, lowest_local_idx = torch.topk(nonzero_probs, k, largest=False)
    lowest_global_idx = nonzero_indices[lowest_local_idx]  # map back to original indices
    
    # Renormalize and sample
    renorm_probs = lowest_vals / lowest_vals.sum()
    chosen_local = torch.multinomial(renorm_probs, num_samples=1)
    
    return lowest_global_idx[chosen_local].item()

def plot_aa_probabilities(probs_1d, standard_aa, alphabet, sampled_idx, n=5, position=None):
    """
    Plots amino acid probabilities, highlighting the n lowest non-zero
    candidates and marking the sampled token.
    
    Args:
        probs_1d: 1D probability tensor (already filtered to standard AAs)
        standard_aa: list of standard AA token indices
        alphabet: ESM alphabet object
        sampled_idx: the token index that was sampled
        n: number of lowest candidates used for sampling
        position: optional sequence position label
    """
    # Get probs only for standard AAs
    aa_names = [alphabet.get_tok(i) for i in standard_aa]
    aa_probs = probs_1d[standard_aa].cpu().numpy()

    # Find the n lowest non-zero among standard AAs
    nonzero_mask = aa_probs > 0
    nonzero_local = nonzero_mask.nonzero()[0]
    nonzero_probs = aa_probs[nonzero_local]
    k = min(n, len(nonzero_probs))
    lowest_local_within_nonzero = nonzero_probs.argsort()[:k]
    lowest_local_idx = nonzero_local[lowest_local_within_nonzero]  # indices in aa_names

    # Map sampled_idx (global) back to local AA index
    sampled_local = standard_aa.index(sampled_idx) if sampled_idx in standard_aa else None

    # Colors: default=steelblue, candidate=orange, sampled=red
    colors = []
    for i in range(len(aa_names)):
        if i == sampled_local:
            colors.append('#e63946')       # sampled — red
        elif i in lowest_local_idx:
            colors.append('#f4a261')       # candidate pool — orange
        else:
            colors.append('#457b9d')       # others — blue

    # Sort by probability for readability
    sorted_order = aa_probs.argsort()[::-1]
    sorted_names = [aa_names[i] for i in sorted_order]
    sorted_probs = aa_probs[sorted_order]
    sorted_colors = [colors[i] for i in sorted_order]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(sorted_names, sorted_probs, color=sorted_colors, edgecolor='black', linewidth=0.7)

    # Annotate bars with prob values
    for bar, prob in zip(bars, sorted_probs):
        if prob > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{prob:.3f}', ha='center', va='bottom', fontsize=7.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e63946', edgecolor='black', label=f'Sampled: {alphabet.get_tok(sampled_idx)}'),
        Patch(facecolor='#f4a261', edgecolor='black', label=f'Candidate pool (top-{n} lowest)'),
        Patch(facecolor='#457b9d', edgecolor='black', label='Other AAs'),
    ]
    ax.legend(handles=legend_elements, fontsize=10)

    title = f'AA Probabilities at position {position}' if position else 'AA Probabilities'
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Amino Acid', fontsize=11)
    ax.set_ylabel('Probability', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def sample_from_lowest_nonzero(probs, n=5):
    probs = probs.squeeze(0)
    nonzero_mask = probs > 0
    nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0]
    nonzero_probs = probs[nonzero_indices]
    k = min(n, len(nonzero_probs))
    lowest_vals, lowest_local_idx = torch.topk(nonzero_probs, k, largest=False)
    lowest_global_idx = nonzero_indices[lowest_local_idx]
    renorm_probs = lowest_vals / lowest_vals.sum()
    chosen_local = torch.multinomial(renorm_probs, num_samples=1)
    return lowest_global_idx[chosen_local].item()


def complete_mask(input_sequence, posi, temperature=1.0, plot=True):
    standard_aa = [alphabet.get_idx(aa) for aa in ['A', 'R', 'N', 'D', 'C', 'Q',
                                                    'E', 'G', 'H', 'I', 'L', 'K',
                                                    'M', 'F', 'P', 'S', 'T', 'W',
                                                    'Y', 'V']]
    data = [("protein1", insert_mask(input_sequence, posi, mask="<mask>"))]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        token_probs = model(batch_tokens, repr_layers=[33])["logits"]

    token_probs /= temperature
    softmax = torch.nn.Softmax(dim=-1)
    probabilities = softmax(token_probs)

    mask_idx = (batch_tokens == alphabet.mask_idx).nonzero(as_tuple=True)

    for token_idx in range(probabilities.size(-1)):
        if token_idx not in standard_aa:
            probabilities[:, :, token_idx] = 0.0

    probs_at_mask = probabilities[mask_idx]  # shape: (1, vocab_size)
    predicted_token = sample_from_lowest_nonzero(probs_at_mask, n=5)

    # Plot BEFORE replacing the token
    if plot:
        plot_aa_probabilities(
            probs_1d=probs_at_mask.squeeze(0),
            standard_aa=standard_aa,
            alphabet=alphabet,
            sampled_idx=predicted_token,
            n=5,
            position=posi
        )

    batch_tokens[mask_idx] = predicted_token
    predicted_residues = [alphabet.get_tok(pred.item()) for pred in batch_tokens[0]]
    seq_predicted = ''.join(predicted_residues[1:-1])

    if input_sequence != seq_predicted:
        print("Mutation added!! 😉")

    return seq_predicted
# 🟠 Laranja — os 5 menores não-zero (candidatos ao sorteio)
# 🔴 Vermelho — o AA efetivamente sorteado


def complete_mask(input_sequence, posi, temperature=1.0, plot = False):

    # Consider only Amino acids tokens for infilling
    standard_aa = [alphabet.get_idx(aa) for aa in ['A', 'R', 'N', 'D', 'C', 'Q', 
                                                   'E', 'G', 'H', 'I', 'L', 'K', 
                                                   'M', 'F', 'P', 'S', 'T', 'W', 
                                                   'Y', 'V']]

    data = [
        ("protein1", insert_mask(input_sequence, posi, mask="<mask>"))]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Predict masked tokens
    with torch.no_grad():
        token_probs = model(batch_tokens, repr_layers=[33])["logits"]

    # Apply temperature
    token_probs /= temperature

    softmax = torch.nn.Softmax(dim=-1)
    probabilities = softmax(token_probs)

    # Get the index of the <mask> token
    mask_idx = (batch_tokens == alphabet.mask_idx).nonzero(as_tuple=True)

        # Zero out probabilities for excluded tokens
        
    for token_idx in range(probabilities.size(-1)):
        if token_idx not in standard_aa:
            probabilities[:, :, token_idx] = 0.0

    # Sample from the probability distribution
    #predicted_tokens = torch.multinomial(probabilities[mask_idx], num_samples=1).squeeze(-1)
    predicted_tokens = sample_from_lowest_nonzero(probabilities[mask_idx])
    probs_at_mask = probabilities[mask_idx]  # shape: (1, vocab_size)
    predicted_tokens = sample_from_lowest_nonzero(probs_at_mask, n=5)

    if plot:
      plot_aa_probabilities(
          probs_1d=probs_at_mask.squeeze(0),
          standard_aa=standard_aa,
          alphabet=alphabet,
          sampled_idx=predicted_tokens,
          n=5,
          position=posi
      )

    # Replace the <mask> token with the predicted token
    batch_tokens[mask_idx] = predicted_tokens

    predicted_residues = [alphabet.get_tok(pred.item()) for pred in batch_tokens[0]]

    seq_predicted = ''.join(predicted_residues[1:-1])

    if input_sequence != seq_predicted:
        print("Mutation added!! 😉")

    return seq_predicted

def generate_sequence(sequence, list_pos, temperature=1.5):

    # shuffle works in-place and returns None, so we create a copy and shuffle it.
    list_pos_copy = list_pos.copy()
    shuffle(list_pos_copy)
    new_sequence = sequence
    for pos in list_pos_copy:
        new_sequence = complete_mask(input_sequence=new_sequence, posi=pos, temperature=temperature)
    return new_sequence


df_reloaded = pd.read_csv('catalytic_sites.csv')
df_reloaded['ACT_SITE_list'] = df_reloaded['ACT_SITE_list'].apply(lambda x: list(map(int, x.split(';'))))
df_reloaded["Decoy_sequence"] = "eita"


for index, row in tqdm(df_reloaded.iterrows(), total=len(df_reloaded), desc="Generating decoys"):
    list_right_indexes = [i-1 for i in row["ACT_SITE_list"]]
    new_decoy = generate_sequence(row["Sequence"], list_right_indexes, temperature=1)
    df_reloaded.at[index, "Decoy_sequence"] = new_decoy

df_reloaded['ACT_SITE_list'] = df_reloaded['ACT_SITE_list'].apply(lambda x: ';'.join(map(str, x)))
df_reloaded.to_csv('catalytic_sites_w_decoys.csv', index=False)