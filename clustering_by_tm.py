#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 19:34:41 2026

@author: joao
"""

import numpy as np
import pandas as pd
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from Bio.PDB import PDBParser
from tmtools import tm_align
from tmtools.io import get_residue_data


# ==============================================================================
# CONFIGURATION — ajuste os caminhos conforme sua estrutura de pastas
# ==============================================================================

PDB_DIR   = "/home/joao/Documentos/Doutorado/Decoys_swissprot/PDB_UniproID"     # PDBs baixados do RCSB
AFDB_DIR  = "/home/joao/Documentos/Doutorado/Decoys_swissprot/AFDB_PDBs" # PDBs baixados do AlphaFold DB
EC_TARGET = "5.3.1.6"   # EC number de interesse — altere aqui
SOURCE    = "afdb"        # "pdb" ou "afdb"

# ==============================================================================


def get_structure_paths(ec_number: str, source: str, df: pd.DataFrame) -> list[str]:
    """
    Dado um EC number e uma fonte ("pdb" ou "afdb"), retorna a lista de caminhos
    .pdb existentes no diretório correspondente para as entradas do DataFrame.

    Args:
        ec_number: EC number a filtrar (ex.: "5.3.1.6").
        source:    "pdb" para RCSB ou "afdb" para AlphaFold DB.
        df:        DataFrame com colunas "EC number", "PDB" e "AlphaFoldDB".

    Returns:
        Lista de caminhos de arquivo válidos (apenas os que existem no disco).
    """
    source = source.lower()
    if source not in ("pdb", "afdb"):
        raise ValueError(f"source deve ser 'pdb' ou 'afdb', recebeu: '{source}'")

    struct_dir = PDB_DIR if source == "pdb" else AFDB_DIR
    col        = "PDB"   if source == "pdb" else "AlphaFoldDB"

    # Filtra entradas com o EC alvo que possuem a coluna de estrutura preenchida
    subset = df[
        df["EC number"].str.contains(ec_number, na=False, regex=False) &
        df[col].notna()
    ].copy()

    if subset.empty:
        print(f"Nenhuma entrada encontrada para EC {ec_number} com fonte {source}.")
        return []

    paths = []
    for _, row in subset.iterrows():
        # Pega o primeiro ID da lista (separada por ";" ou ",")
        raw_ids = str(row[col]).replace(",", ";").split(";")
        struct_id = raw_ids[0].strip()

        if not struct_id:
            continue

        candidate = os.path.join(struct_dir, f"{struct_id}.pdb")
        if os.path.exists(candidate):
            paths.append(candidate)
        else:
            print(f"  [aviso] Arquivo não encontrado: {candidate}")

    print(f"\nEstruturas encontradas para EC {ec_number} ({source.upper()}): {len(paths)}")
    return paths


def cache_structures(pdb_paths: list[str]) -> dict:
    """
    Pré-carrega todas as estruturas PDB em memória, extraindo coordenadas e sequência
    da primeira cadeia de cada arquivo.

    Args:
        pdb_paths: Lista de caminhos para arquivos .pdb.

    Returns:
        Dicionário {pdb_id: (coords, seq)} para estruturas carregadas com sucesso.
    """
    parser = PDBParser(QUIET=True)
    cache  = {}

    print("Pré-carregando estruturas...")
    for path in tqdm(pdb_paths, desc="Caching"):
        pdb_id = os.path.basename(path)[:-4]
        try:
            structure = parser.get_structure(pdb_id, path)[0]
            chain     = next(structure.get_chains())
            coords, seq = get_residue_data(chain)
            cache[pdb_id] = (coords, seq)
        except Exception as e:
            print(f"  [erro] {pdb_id}: {e}")

    print(f"Estruturas carregadas com sucesso: {len(cache)}")
    return cache


def compute_tm_matrix(cache: dict) -> pd.DataFrame:
    """
    Calcula a matriz de TM-scores para todas as combinações de estruturas em cache.

    Args:
        cache: Dicionário {pdb_id: (coords, seq)}.

    Returns:
        DataFrame n×n com TM-scores (normalizado pela cadeia 1).
    """
    ids = list(cache.keys())
    n   = len(ids)

    tm_matrix = np.eye(n)  # diagonal = 1.0 (auto-comparação)

    print(f"Calculando {n * n} alinhamentos...")
    for i in tqdm(range(n), desc="Matriz TM-score"):
        coords1, seq1 = cache[ids[i]]
        for j in range(n):
            if i == j:
                continue
            coords2, seq2 = cache[ids[j]]
            try:
                result = tm_align(coords1, coords2, seq1, seq2)
                tm_matrix[i, j] = result.tm_norm_chain1
            except Exception as e:
                print(f"  [erro] {ids[i]} vs {ids[j]}: {e}")
                tm_matrix[i, j] = np.nan

    return pd.DataFrame(tm_matrix, index=ids, columns=ids)


def plot_clustermap(df_tm: pd.DataFrame, ec_number: str, source: str,
                    output_path: str | None = None) -> None:
    """
    Gera e exibe (e opcionalmente salva) um clustermap hierárquico dos TM-scores.

    Args:
        df_tm:       DataFrame n×n de TM-scores.
        ec_number:   EC number usado (para título e nome do arquivo).
        source:      Fonte das estruturas ("pdb" ou "afdb").
        output_path: Caminho para salvar a figura. Se None, gera um nome automático.
    """
    if df_tm.empty or df_tm.shape[0] < 2:
        print("Matriz insuficiente para clusterização (mínimo 2 estruturas).")
        return

    annotate = df_tm.shape[0] < 15  # anota valores apenas para matrizes pequenas

    g = sns.clustermap(
        df_tm.fillna(0),
        cmap="magma",
        vmin=0,
        vmax=1,
        metric="euclidean",
        method="ward",
        figsize=(12, 12),
        annot=annotate,
        fmt=".2f" if annotate else "",
    )

    ec_safe  = ec_number.replace(".", "_")
    title    = f"TM-score Clustermap — EC {ec_number} ({source.upper()})"
    g.fig.suptitle(title, y=1.02, fontsize=14)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")

    if output_path is None:
        output_path = f"clustermap_EC{ec_safe}_{source}.png"

    g.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Clustermap salvo em: {output_path}")
    plt.show()


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

processed_df_act = pd.read_csv("catalytic_sites.csv")
# 1. Seleciona as estruturas para o EC e fonte desejados
pdb_paths = get_structure_paths(EC_TARGET, SOURCE, processed_df_act)

if pdb_paths:
    # 2. Carrega estruturas em memória
    data_cache = cache_structures(pdb_paths)

    # 3. Calcula matriz de TM-scores
    df_tm = compute_tm_matrix(data_cache)

    # 4. Salva a matriz
    ec_safe  = EC_TARGET.replace(".", "_")
    csv_path = f"tm_matrix_EC{ec_safe}_{SOURCE}.csv"
    df_tm.to_csv(csv_path)
    print(f"Matriz salva em: {csv_path}")

    # 5. Gera o clustermap
    plot_clustermap(df_tm, EC_TARGET, SOURCE)
else:
    print("Nenhuma estrutura disponível — verifique os caminhos e o EC number.")