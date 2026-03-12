#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:54:46 2026

@author: joao
"""

import requests
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Config ───────────────────────────────────────────────
CSV_PATH   = "catalytic_sites.csv"
UNIPROT_COL = "Entry"          # <- mude se necessário
OUTPUT_DIR  = "AFDB_PDBs"
FMT         = "pdb"            # "pdb" ou "cif"
MAX_WORKERS = 8                # downloads paralelos

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Session com retry automático ────────────────────────
def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session

# ── Funções principais ───────────────────────────────────
def get_pdb_url(session: requests.Session, uniprot_id: str, fmt: str = "pdb") -> str:
    """Obtém a URL do arquivo estrutural via API do AlphaFold DB."""
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    resp = session.get(api_url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise ValueError(f"Nenhum dado retornado para {uniprot_id}")
    entry = data[0]  # primeiro fragmento (F1)
    return entry["pdbUrl"] if fmt == "pdb" else entry["cifUrl"]


def download_one(uniprot_id: str, fmt: str = "pdb") -> tuple[str, str]:
    """
    Baixa o arquivo estrutural de um UniProt ID.
    Retorna (uniprot_id, status) onde status é "ok", "skipped" ou mensagem de erro.
    """
    ext = ".pdb" if fmt == "pdb" else ".cif"
    out_path = os.path.join(OUTPUT_DIR, f"{uniprot_id}{ext}")

    # Pula se já existe
    if os.path.exists(out_path):
        return uniprot_id, "skipped"

    session = make_session()
    try:
        file_url = get_pdb_url(session, uniprot_id, fmt)
        resp = session.get(file_url, timeout=60)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
        return uniprot_id, "ok"
    except requests.HTTPError as e:
        return uniprot_id, f"HTTP {e.response.status_code}"
    except Exception as e:
        return uniprot_id, f"ERRO: {e}"


# ── Leitura do DataFrame ─────────────────────────────────
df = pd.read_csv(CSV_PATH)
uniprot_ids = (
    df[UNIPROT_COL]
    .dropna()
    .str.strip()
    .unique()
    .tolist()
)
print(f"Total de IDs únicos: {len(uniprot_ids)}")

# ── Download paralelo com barra de progresso ─────────────
ok_count      = 0
skipped_count = 0
failed        = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(download_one, uid, FMT): uid for uid in uniprot_ids}

    with tqdm(total=len(futures), desc="Baixando PDBs", unit="struct") as pbar:
        for future in as_completed(futures):
            uid, status = future.result()
            if status == "ok":
                ok_count += 1
            elif status == "skipped":
                skipped_count += 1
            else:
                failed.append((uid, status))
            pbar.update(1)
            pbar.set_postfix(ok=ok_count, skip=skipped_count, fail=len(failed))

# ── Relatório final ──────────────────────────────────────
print(f"\n✅ Baixados   : {ok_count}")
print(f"⏭  Já existiam: {skipped_count}")
print(f"❌ Falhas      : {len(failed)}")

if failed:
    fail_df = pd.DataFrame(failed, columns=["UniProt_ID", "Motivo"])
    fail_path = os.path.join(OUTPUT_DIR, "failed_downloads.csv")
    fail_df.to_csv(fail_path, index=False)
    print(f"\nIDs com falha salvos em: {fail_path}")
    print(fail_df.to_string(index=False))