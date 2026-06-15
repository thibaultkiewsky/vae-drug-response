"""
hp_search.py — Grid search d'hyperparamètres VAE moléculaire
=============================================================
Pour chaque combinaison d'hyperparamètres :
  1. Entraîne le VAE sur les fingerprints (200k molécules)
  2. Encode les molécules PRISM
  3. Calcule les proximités latentes vs proximités d'effet (AUC)
  4. Sauvegarde les poids + le graphe + un CSV de résultats

Usage :
    python hp_search.py \
        --fingerprints data/data_mol/processed/all_fingerprints.csv \
        --prism        secondary-screen-dose-response-curve-parameters.csv \
        --outdir       results/hp_search
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import argparse
import itertools
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from scipy.stats import spearmanr
from itertools import combinations
from tqdm import tqdm

from vae import VAE, KLAnnealingCallback
from smiles_to_fp import smiles_to_fp

# =============================================================================
#  Grille d'hyperparamètres 
# =============================================================================

HP_GRID = {
    "latent_dim"     : [128],
    "encoder_layers" : [[1024, 512, 256, 128]],
    "lambda_kl_max"  : [1e-2],
    "dropout_rate"   : [0.2],
    # paramètres fixes 
    # "learning_rate": [1e-3],
    # "kl_warmup_epochs": [5, 10],
}

# Paramètres fixes pour tous les runs
FIXED = {
    "input_dim"        : 2048,
    "activation"       : "relu",
    "use_batch_norm"   : True,
    "output_activation": "sigmoid",
    "learning_rate"    : 1e-3,
    "batch_size"       : 256,
    "epochs"           : 50,
    "kl_warmup_epochs" : 5,
}

# Seuil minimum de lignées communes pour calculer la corrélation d'effet
MIN_CELL_LINES = 5

# =============================================================================
#  Utilitaires
# =============================================================================

def load_prism(path):
    """Charge le CSV PRISM et retourne un DataFrame pivoté (smiles × lignées)."""
    print("Chargement PRISM...")
    df = pd.read_csv(path, low_memory=False)
    df = df[["smiles", "depmap_id", "auc"]].dropna()
    df = df.groupby(["smiles", "depmap_id"], as_index=False)["auc"].mean()
    pivot = df.pivot(index="smiles", columns="depmap_id", values="auc")
    print(f"  {len(pivot)} molécules, {pivot.shape[1]} lignées cellulaires")
    return pivot


def encode_batch(vae, fps, batch_size):
    """Encode un array de fingerprints → z_mean (numpy)."""
    parts = []
    for i in range(0, len(fps), batch_size):
        z_mean, _, _ = vae.encode(fps[i : i + batch_size])
        parts.append(z_mean.numpy())
    return np.concatenate(parts, axis=0)


def compute_pairs(Z, AUC, min_cells=MIN_CELL_LINES):
    """
    Calcule pour toutes les paires :
      - distance euclidienne entre z_mean
      - corrélation de Spearman entre profils AUC
    Retourne (lat_dists, eff_corrs) filtrés (sans NaN).
    """
    n = len(Z)
    lat_dists, eff_corrs = [], []

    for i, j in combinations(range(n), 2):
        d = float(np.linalg.norm(Z[i] - Z[j]))
        lat_dists.append(d)

        a_i, a_j = AUC[i], AUC[j]
        mask = ~(np.isnan(a_i) | np.isnan(a_j))
        if mask.sum() < min_cells:
            eff_corrs.append(np.nan)
        else:
            rho, _ = spearmanr(a_i[mask], a_j[mask])
            eff_corrs.append(rho)

    lat_dists = np.array(lat_dists)
    eff_corrs = np.array(eff_corrs)
    valid = ~np.isnan(eff_corrs)
    return lat_dists[valid], eff_corrs[valid]


def save_graph(lat_dists, eff_corrs, spearman_r, spearman_p,
               pearson_r, pearson_p, run_name, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [3, 1]})
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")

    ax = axes[0]
    hb = ax.hexbin(lat_dists, eff_corrs, gridsize=60, cmap="plasma",
                   mincnt=1, linewidths=0.2)
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("Nombre de paires", color="white", fontsize=10)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    z_fit  = np.polyfit(lat_dists, eff_corrs, 1)
    x_line = np.linspace(lat_dists.min(), lat_dists.max(), 300)
    ax.plot(x_line, np.polyval(z_fit, x_line),
            color="#00e5ff", lw=2, ls="--", label="Tendance", zorder=5)

    ax.set_xlabel("Distance euclidienne latente (entre μ)", color="white", fontsize=11)
    ax.set_ylabel("Similarité d'effet (Spearman AUC)", color="white", fontsize=11)
    ax.set_title(run_name, color="white", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    txt = (f"Pearson   r = {pearson_r:+.3f}  (p={pearson_p:.1e})\n"
           f"Spearman  ρ = {spearman_r:+.3f}  (p={spearman_p:.1e})\n"
           f"N paires = {len(lat_dists):,}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=10, color="#e6edf3",
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="#21262d", edgecolor="#30363d", alpha=0.9))
    ax.legend(facecolor="#21262d", edgecolor="#30363d",
              labelcolor="white", fontsize=10)

    ax2 = axes[1]
    ax2.hist(lat_dists, bins=60, color="#7c3aed", edgecolor="#0d1117",
             alpha=0.85, orientation="horizontal")
    ax2.set_xlabel("Nombre de paires", color="white", fontsize=10)
    ax2.set_ylabel("Distance latente", color="white", fontsize=10)
    ax2.set_title("Distribution\ndes distances", color="white", fontsize=11, pad=8)
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")

    plt.tight_layout(pad=2)
    path = os.path.join(outdir, f"{run_name}.png")
    plt.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return path

# =============================================================================
#  Boucle principale
# =============================================================================

def run_search(fingerprints_path, prism_path, outdir):
    os.makedirs(outdir, exist_ok=True)

    # ── Chargement des données une seule fois ─────────────────────────────────

    print("Chargement des fingerprints d'entraînement...")
    df_fp = pd.read_csv(fingerprints_path)
    train_fps = df_fp.iloc[:, 2:].to_numpy(dtype=np.float32)
    print(f"  {len(train_fps)} molécules, {train_fps.shape[1]} bits")

    df_auc = load_prism(prism_path)

    print("Calcul des fingerprints PRISM...")
    smiles_list = df_auc.index.tolist()
    prism_fps, valid_idx = smiles_to_fp(smiles_list)
    prism_fps = prism_fps.astype(np.float32)
    df_auc_valid = df_auc.iloc[valid_idx].reset_index(drop=True)
    AUC = df_auc_valid.values
    print(f"  {len(prism_fps)} molécules PRISM valides")

    # ── Génération de toutes les combinaisons ─────────────────────────────────

    keys   = list(HP_GRID.keys())
    values = list(HP_GRID.values())
    combos = list(itertools.product(*values))
    print(f"\n{len(combos)} combinaisons à tester\n")

    results = []

    for combo_idx, combo in enumerate(combos):
        hp = dict(zip(keys, combo))
        hp.update(FIXED)

        # Nom lisible pour ce run
        run_name = (
            f"z{hp['latent_dim']}"
            f"_enc{'x'.join(str(u) for u in hp['encoder_layers'])}"
            f"_kl{hp['lambda_kl_max']:.0e}"
            f"_drop{hp['dropout_rate']}"
        )
        print(f"\n{'='*60}")
        print(f"Run {combo_idx+1}/{len(combos)} : {run_name}")
        print(f"{'='*60}")

        run_dir = os.path.join(outdir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        # Sauvegarde des hyperparamètres
        with open(os.path.join(run_dir, "hparams.json"), "w") as f:
            json.dump({k: (v if not isinstance(v, list) else v)
                       for k, v in hp.items()}, f, indent=2)

        # ── Entraînement ──────────────────────────────────────────────────────

        vae = VAE(
            input_dim         = hp["input_dim"],
            latent_dim        = hp["latent_dim"],
            encoder_layers    = hp["encoder_layers"],
            decoder_layers    = None,   # miroir automatique
            activation        = hp["activation"],
            dropout_rate      = hp["dropout_rate"],
            use_batch_norm    = hp["use_batch_norm"],
            output_activation = hp["output_activation"],
            lambda_kl         = 0.0,    # part de 0, le callback monte jusqu'à lambda_kl_max
            name              = "vae_mol"
        )
        vae.build((None, hp["input_dim"]))
        vae.compile(optimizer=keras.optimizers.Adam(hp["learning_rate"]))

        kl_cb = KLAnnealingCallback(
            lambda_kl_max    = hp["lambda_kl_max"],
            kl_warmup_epochs = hp["kl_warmup_epochs"]
        )

        history = vae.fit(
            train_fps,
            epochs     = hp["epochs"],
            batch_size = hp["batch_size"],
            callbacks  = [kl_cb],
            verbose    = 1,
        )

        # Sauvegarde des poids
        weights_path = os.path.join(run_dir, "weights.weights.h5")
        vae.save_weights(weights_path)
        print(f"  Poids sauvegardés → {weights_path}")

        # Sauvegarde de la courbe de loss
        fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
        fig_loss.patch.set_facecolor("#0d1117")
        ax_loss.set_facecolor("#161b22")
        #ax_loss.plot(history.history["loss"], color="#00e5ff", label="total loss")
        ax_loss.plot(history.history["reconstruction_loss"], color="#7c3aed",
                     label="reconstruction")
        #ax_loss.plot(history.history["kl_loss"], color="#f97316", label="KL")
        ax_loss.set_title(f"Loss — {run_name}", color="white")
        ax_loss.set_xlabel("Époque", color="white")
        ax_loss.tick_params(colors="white")
        for spine in ax_loss.spines.values():
            spine.set_edgecolor("#405977")
        ax_loss.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "loss.png"), dpi=120,
                    bbox_inches="tight", facecolor=fig_loss.get_facecolor())
        plt.close()

        # ── Validation : proximité latente vs proximité d'effet ───────────────

        print("  Encodage des molécules PRISM...")
        Z = encode_batch(vae, prism_fps, hp["batch_size"])

        print(f"  Calcul des paires ({len(Z)*(len(Z)-1)//2:,} paires)...")
        lat_dists, eff_corrs = compute_pairs(Z, AUC)

        from scipy.stats import pearsonr
        spearman_r, spearman_p = spearmanr(lat_dists, eff_corrs)
        pearson_r,  pearson_p  = pearsonr(lat_dists,  eff_corrs)

        print(f"  Spearman ρ = {spearman_r:+.4f}  (p={spearman_p:.2e})")
        print(f"  Pearson  r = {pearson_r:+.4f}  (p={pearson_p:.2e})")

        # Graphe
        graph_path = save_graph(
            lat_dists, eff_corrs,
            spearman_r, spearman_p,
            pearson_r,  pearson_p,
            run_name, run_dir
        )
        print(f"  Graphe sauvegardé → {graph_path}")

        # Résultat de ce run
        results.append({
            "run"            : run_name,
            "latent_dim"     : hp["latent_dim"],
            "encoder_layers" : str(hp["encoder_layers"]),
            "lambda_kl_max"  : hp["lambda_kl_max"],
            "dropout_rate"   : hp["dropout_rate"],
            "spearman_rho"   : round(spearman_r, 4),
            "spearman_p"     : spearman_p,
            "pearson_r"      : round(pearson_r, 4),
            "pearson_p"      : pearson_p,
            "n_pairs"        : len(lat_dists),
            "final_loss"     : round(history.history["loss"][-1], 4),
            "weights_path"   : weights_path,
        })

        # Sauvegarde incrémentale du CSV (utile si le script est interrompu)
        pd.DataFrame(results).sort_values("spearman_rho", ascending=False).to_csv(
            os.path.join(outdir, "results.csv"), index=False
        )

    # ── Résumé final ──────────────────────────────────────────────────────────

    df_res = pd.DataFrame(results).sort_values("spearman_rho", ascending=False)
    print("\n" + "="*60)
    print("RÉSULTATS (triés par Spearman ρ décroissant)")
    print("="*60)
    print(df_res[["run", "spearman_rho", "pearson_r", "final_loss"]].to_string(index=False))
    print(f"\nMeilleur run : {df_res.iloc[0]['run']}")
    print(f"  Spearman ρ = {df_res.iloc[0]['spearman_rho']}")
    print(f"  Poids      : {df_res.iloc[0]['weights_path']}")

    return df_res


# =============================================================================
#  CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fingerprints", required=True,
                        help="CSV des fingerprints d'entraînement (200k molécules)")
    parser.add_argument("--prism", required=True,
                        help="CSV PRISM secondary screen")
    parser.add_argument("--outdir", default="results/hp_search",
                        help="Dossier de sortie")
    args = parser.parse_args()

    run_search(args.fingerprints, args.prism, args.outdir)