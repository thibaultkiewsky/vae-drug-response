"""
train_vae_cell.py — Entraînement du VAE génomique sur TCGA + validation phénotypique
======================================================================================
Modifie la section CONFIG ci-dessous pour changer les hyperparamètres.

Usage :
  python train_vae_cell.py
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import keras
import tensorflow as tf

from vae import VAE, KLAnnealingCallback

Umap = True

if Umap:
    import umap
    HAS_UMAP = True
else:
    HAS_UMAP = False
    from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder


# =============================================================================
#  CONFIG — tout modifier ici
# =============================================================================

DATA = dict(
    datadir     = "data/tcga_processed",   # dossier produit par prepare_tcga.py
    outdir      = "results/vae_cell",
)

VAE_HP = dict(
    latent_dim      = 64,
    encoder_layers  = [2048, 1024, 256],   # couches cachées encodeur
    decoder_layers  = None,                 # None = miroir de l'encodeur
    activation      = "relu",
    dropout_rate    = 0.2,
    use_batch_norm  = True,
    output_activation = "linear",          # linear car expression continue
)

TRAIN = dict(
    epochs          = 50,
    batch_size      = 128,
    learning_rate   = 1e-3,
    lambda_kl_max   = 1e-4,
    kl_warmup       = 10,                  # époques pour atteindre lambda_kl_max
)

# Colonnes phénotypiques à visualiser (dans l'ordre de priorité)
PHENO_COLS = [
    "cancer type abbreviation",
    "_primary_disease",
    "_gender",
    "gender",
    "_race",
    "race",
    "_sample_type",
    "sample_type",
]

# =============================================================================
#  VAECell — surcharge de train_step pour utiliser MSE (expression continue)
# =============================================================================

class VAECell(VAE):
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data, training=True)
            reconstruction       = self.decoder(z, training=True)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - reconstruction), axis=1)
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.lambda_kl * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss":                self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss":             self.kl_loss_tracker.result(),
        }


# =============================================================================
#  Visualisation
# =============================================================================

def get_2d_embedding(Z):
    if HAS_UMAP:
        return umap.UMAP(n_components=2, random_state=42,
                         n_neighbors=30, min_dist=0.1).fit_transform(Z)
    return PCA(n_components=2, random_state=42).fit_transform(Z)


def plot_embedding(Z2d, labels, title, output_path, max_categories=30):
    unique = [l for l in pd.Series(labels).unique() if pd.notna(l)]
    if len(unique) > max_categories:
        unique = pd.Series(labels).value_counts().head(max_categories).index.tolist()

    cmap   = cm.get_cmap("tab20", max(len(unique), 1))
    colors = {cat: cmap(i) for i, cat in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    mask_other = ~pd.Series(labels).isin(unique)
    if mask_other.any():
        ax.scatter(Z2d[mask_other, 0], Z2d[mask_other, 1],
                   c="#444", s=4, alpha=0.3, label="other")
    for cat in unique:
        mask = np.array([l == cat for l in labels])
        ax.scatter(Z2d[mask, 0], Z2d[mask, 1],
                   c=[colors[cat]], s=8, alpha=0.7, label=str(cat))

    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("dim 1", color="white")
    ax.set_ylabel("dim 2", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.legend(markerscale=2, fontsize=7, loc="upper right", ncol=2,
              facecolor="#21262d", edgecolor="#30363d", labelcolor="white")
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  → {output_path}")


def plot_loss(history, outdir):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.plot(history.history["loss"],                color="#00e5ff", label="total")
    ax.plot(history.history["reconstruction_loss"], color="#7c3aed", label="reconstruction")
    ax.plot(history.history["kl_loss"],             color="#f97316", label="KL")
    ax.set_xlabel("Époque", color="white")
    ax.set_title("Courbe de loss — VAE génomique", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss.png"), dpi=130,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


# =============================================================================
#  Pipeline
# =============================================================================

def main():
    outdir = DATA["outdir"]
    os.makedirs(outdir, exist_ok=True)

    # ── Chargement ────────────────────────────────────────────────────────────
    print("Chargement des données...")
    X = np.load(os.path.join(DATA["datadir"], "expression.npy")).astype(np.float32)
    print(f"  Expression : {X.shape}")

    with open(os.path.join(DATA["datadir"], "sample_ids.txt")) as f:
        sample_ids = [l.strip() for l in f]

    pheno_path = os.path.join(DATA["datadir"], "phenotype_aligned.tsv")
    pheno = pd.read_csv(pheno_path, sep="\t", index_col=0) if os.path.exists(pheno_path) else None
    if pheno is not None:
        pheno.index = pheno.index.str[:15]
        print(f"  Phénotypes : {pheno.shape}")

    # ── Modèle ────────────────────────────────────────────────────────────────
    vae = VAECell(
        input_dim = X.shape[1],
        **VAE_HP,
        lambda_kl = 0.0,
        name      = "vae_cell",
    )
    vae.build((None, X.shape[1]))
    vae.compile(optimizer=keras.optimizers.Adam(TRAIN["learning_rate"]))
    vae.summary_full()

    # ── Entraînement ──────────────────────────────────────────────────────────
    print(f"\nEntraînement ({TRAIN['epochs']} époques)...")
    history = vae.fit(
        X,
        epochs     = TRAIN["epochs"],
        batch_size = TRAIN["batch_size"],
        callbacks  = [KLAnnealingCallback(TRAIN["lambda_kl_max"], TRAIN["kl_warmup"])],
        verbose    = 1,
    )
    weights_path = os.path.join(outdir, "vae_cell_weights.weights.h5")
    vae.save_weights(weights_path)
    print(f"Poids sauvegardés → {weights_path}")
    plot_loss(history, outdir)

    # ── Encodage ──────────────────────────────────────────────────────────────
    print("Encodage dans l'espace latent...")
    Z = np.concatenate([
        vae.encode(X[i : i + TRAIN["batch_size"]])[0].numpy()
        for i in range(0, len(X), TRAIN["batch_size"])
    ], axis=0)
    np.save(os.path.join(outdir, "latent_z.npy"), Z)
    print(f"  Z shape : {Z.shape}")

    # ── UMAP / PCA ────────────────────────────────────────────────────────────
    method = "UMAP" if HAS_UMAP else "PCA"
    print(f"Réduction 2D ({method})...")
    Z2d = get_2d_embedding(Z)
    np.save(os.path.join(outdir, "Z2d.npy"), Z2d)

    # ── Visualisations phénotypiques ──────────────────────────────────────────
    if pheno is not None:
        sample_ids_short = [s[:15] for s in sample_ids]
        pheno_aligned    = pheno.reindex(sample_ids_short)
        available_cols   = [c for c in PHENO_COLS if c in pheno_aligned.columns]

        print(f"\nVisualisations ({len(available_cols)} variables)...")
        scores = {}
        for col in available_cols:
            labels = pheno_aligned[col].fillna("unknown").tolist()
            fname  = col.replace(" ", "_").replace("/", "_")
            plot_embedding(Z2d, labels,
                           title       = f"VAE cell — {col}",
                           output_path = os.path.join(outdir, f"umap_{fname}.png"))
            try:
                le = LabelEncoder()
                y  = le.fit_transform(labels)
                idx = (np.random.choice(len(Z), 5000, replace=False)
                       if len(Z) > 5000 else np.arange(len(Z)))
                sc = silhouette_score(Z[idx], y[idx], metric="euclidean")
                scores[col] = round(float(sc), 4)
                print(f"  Silhouette [{col}] = {sc:.4f}")
            except Exception as e:
                print(f"  Silhouette [{col}] : erreur ({e})")

        pd.DataFrame.from_dict(scores, orient="index",
                               columns=["silhouette"]).to_csv(
            os.path.join(outdir, "phenotype_scores.csv"))

    print(f"\n✅  Terminé. Résultats dans : {outdir}/")
    return vae


if __name__ == "__main__":
    main()