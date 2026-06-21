"""
train_global_model.py — Modèle global : VAE mol + VAE cell + tête MLP → AUC
=============================================================================
Architecture :
    fingerprint (2048) → VAE_mol → z_mol (latent_dim_mol)  ─┐
                                                              ├─ concat → MLP → AUC
    expression  (5806) → VAE_cell → z_cell (latent_dim_cell) ┘

Pipeline :
  1. Charge les poids pré-entraînés des deux VAE
  2. Prépare les données PRISM (couples molécule × lignée → AUC)
  3. Cross-validation aléatoire K-fold :
       - "molécule vue"   : molécule dans le train, dans le test
       - "molécule non vue" : molécule absente du train
       - "cellule vue"    : lignée dans le train, dans le test
       - "cellule non vue": lignée absente du train
  4. Fine-tune les deux VAE + tête MLP ensemble
  5. Résultats : MSE / PCC / Spearman par scénario + graphes

Modifie la section CONFIG ci-dessous pour changer les hyperparamètres.

Usage :
    python train_global_model.py
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold

from vae import VAE, KLAnnealingCallback
from smiles_to_fp import smiles_to_fp


# =============================================================================
#  CONFIG — tout modifier ici
# =============================================================================

PATHS = dict(
    prism        = r"C:\Users\thiba\Documents\travail\PFE\vae-drug-response\data\data_finetune\raw\secondary-screen-dose-response-curve-parameters.csv",
    ccle_expr    = r"C:\Users\thiba\Documents\travail\PFE\vae-drug-response\data\ccle_processed\expression.npy",   # expression CCLE (lignées)
    ccle_ids     = r"C:\Users\thiba\Documents\travail\PFE\vae-drug-response\data\ccle_processed\cell_line_ids.txt",
    mol_weights  = r"C:\Users\thiba\Documents\travail\PFE\vae-drug-response\results\hp_search\z64_enc1024x512x256_kl1e-02_drop0.2\weights.weights.h5",
    cell_weights = r"C:\Users\thiba\Documents\travail\PFE\vae-drug-response\results\vae_cell\vae_cell_weights.weights.h5",
    outdir       = "results/global_model",
)

# Hyperparamètres VAE mol (doivent correspondre aux poids chargés)
VAE_MOL_HP = dict(
    input_dim         = 2048,
    latent_dim        = 64,
    encoder_layers    = [1024, 512, 256],
    decoder_layers    = None,
    activation        = "relu",
    dropout_rate      = 0.2,
    use_batch_norm    = True,
    output_activation = "sigmoid",
    lambda_kl         = 1e-4,
)

# Hyperparamètres VAE cell (doivent correspondre aux poids chargés)
VAE_CELL_HP = dict(
    input_dim         = 5806,
    latent_dim        = 64,
    encoder_layers    = [2048, 1024, 256],
    decoder_layers    = None,
    activation        = "relu",
    dropout_rate      = 0.2,
    use_batch_norm    = True,
    output_activation = "linear",
    lambda_kl         = 1e-4,
)

# Tête MLP de prédiction
MLP_HP = dict(
    hidden_layers = [256, 128, 64],   # couches après la concaténation des z
    dropout_rate  = 0.2,
    activation    = "relu",
)

# Entraînement global
TRAIN = dict(
    epochs        = 10,
    batch_size    = 256,
    learning_rate = 1e-4,             # plus faible pour le finetuning
    n_folds       = 5,
    random_seed   = 42,
)

# Poids des différents termes de la loss totale
LOSS_WEIGHTS = dict(
    prediction = 1.0,    # MSE sur l'AUC (objectif principal)
    kl_mol     = 1e-5,   # régularisation KL du VAE mol (faible pour préserver les poids)
    kl_cell    = 1e-5,   # régularisation KL du VAE cell
)


# =============================================================================
#  Sous-classes VAE (pour charger les poids correctement)
# =============================================================================

class VAECell(VAE):
    """VAE cell avec MSE comme loss de reconstruction."""
    def train_step(self, data):
        # Non utilisé dans le modèle global (le train_step global gère tout)
        raise NotImplementedError


# =============================================================================
#  Modèle global
# =============================================================================

class DrugResponseModel(keras.Model):
    """
    Modèle global : deux encodeurs VAE + tête MLP → prédiction AUC.

    Inputs :
        fp   : fingerprint molécule  (batch, input_dim_mol)
        expr : expression génique    (batch, input_dim_cell)

    Output :
        auc_pred : AUC prédit        (batch, 1)

    La loss totale combine :
        L = w_pred * MSE(auc, auc_pred)
          + w_kl_mol  * KL_mol
          + w_kl_cell * KL_cell
    """

    def __init__(self, vae_mol, vae_cell, mlp_layers, mlp_dropout,
                 mlp_activation, loss_weights, **kwargs):
        super().__init__(**kwargs)

        self.vae_mol    = vae_mol
        self.vae_cell   = vae_cell
        self.loss_weights = loss_weights

        # Tête MLP construite sur la concaténation z_mol + z_cell
        latent_total = vae_mol.latent_dim + vae_cell.latent_dim
        self.mlp = self._build_mlp(latent_total, mlp_layers,
                                   mlp_dropout, mlp_activation)

        # Trackers
        self.loss_tracker      = keras.metrics.Mean(name="loss")
        self.pred_loss_tracker = keras.metrics.Mean(name="pred_loss")
        self.kl_mol_tracker    = keras.metrics.Mean(name="kl_mol")
        self.kl_cell_tracker   = keras.metrics.Mean(name="kl_cell")

    def _build_mlp(self, input_dim, hidden_layers, dropout_rate, activation):
        inp = keras.Input(shape=(input_dim,), name="z_concat")
        x = inp
        for i, units in enumerate(hidden_layers):
            x = layers.Dense(units, name=f"mlp_dense_{i}")(x)
            x = layers.BatchNormalization(name=f"mlp_bn_{i}")(x)
            x = layers.Activation(activation, name=f"mlp_act_{i}")(x)
            x = layers.Dropout(dropout_rate, name=f"mlp_drop_{i}")(x)
        out = layers.Dense(1, activation="sigmoid", name="auc_output")(x)
        return keras.Model(inp, out, name="mlp_head")

    @property
    def metrics(self):
        return [self.loss_tracker, self.pred_loss_tracker,
                self.kl_mol_tracker, self.kl_cell_tracker]

    def call(self, inputs, training=False):
        fp, expr = inputs
        z_mol_mean,  z_mol_logvar,  _ = self.vae_mol.encoder(fp,   training=training)
        z_cell_mean, z_cell_logvar, _ = self.vae_cell.encoder(expr, training=training)
        z = tf.concat([z_mol_mean, z_cell_mean], axis=1)
        return self.mlp(z, training=training)

    def _kl_loss(self, z_mean, z_log_var):
        kl = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        return tf.reduce_mean(tf.reduce_sum(kl, axis=1))

    def train_step(self, data):
        (fp, expr), auc = data

        with tf.GradientTape() as tape:
            z_mol_mean,  z_mol_logvar,  z_mol  = self.vae_mol.encoder(fp,   training=True)
            z_cell_mean, z_cell_logvar, z_cell = self.vae_cell.encoder(expr, training=True)

            z = tf.concat([z_mol_mean, z_cell_mean], axis=1)
            auc_pred = self.mlp(z, training=True)

            pred_loss = tf.reduce_mean(tf.square(auc - auc_pred))
            kl_mol    = self._kl_loss(z_mol_mean,  z_mol_logvar)
            kl_cell   = self._kl_loss(z_cell_mean, z_cell_logvar)

            total_loss = (
                self.loss_weights["prediction"] * pred_loss
                + self.loss_weights["kl_mol"]   * kl_mol
                + self.loss_weights["kl_cell"]  * kl_cell
            )

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.loss_tracker.update_state(total_loss)
        self.pred_loss_tracker.update_state(pred_loss)
        self.kl_mol_tracker.update_state(kl_mol)
        self.kl_cell_tracker.update_state(kl_cell)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (fp, expr), auc = data
        auc_pred = self((fp, expr), training=False)
        pred_loss = tf.reduce_mean(tf.square(auc - auc_pred))
        self.pred_loss_tracker.update_state(pred_loss)
        return {"pred_loss": self.pred_loss_tracker.result()}


# =============================================================================
#  Préparation des données PRISM
# =============================================================================

def load_prism_pairs(prism_path, ccle_expr, ccle_ids):
    """
    Construit les triplets (fingerprint, expression, auc) depuis PRISM.

    Retourne :
        fps        : np.array (N, 2048)   fingerprints
        exprs      : np.array (N, n_genes) expressions
        aucs       : np.array (N,)         AUC cibles
        mol_ids    : list[str]             identifiant molécule (smiles)
        cell_ids   : list[str]             identifiant lignée (depmap_id)
    """
    print("Chargement PRISM...")
    df = pd.read_csv(prism_path, low_memory=False)
    df = df[["smiles", "depmap_id", "auc"]].dropna()
    df = df.groupby(["smiles", "depmap_id"], as_index=False)["auc"].mean()

    # Filtrer sur les lignées disponibles dans CCLE
    ccle_id_list = [l.strip() for l in open(ccle_ids)]
    ccle_id_set  = set(ccle_id_list)
    df = df[df["depmap_id"].isin(ccle_id_set)]
    print(f"  {len(df)} paires après filtre CCLE ({len(ccle_id_list)} lignées)")

    # Fingerprints
    smiles_unique = df["smiles"].unique().tolist()
    print(f"  Calcul des fingerprints ({len(smiles_unique)} molécules uniques)...")
    fps_arr, valid_idx = smiles_to_fp(smiles_unique)
    fps_arr = fps_arr.astype(np.float32)
    valid_smiles = [smiles_unique[i] for i in valid_idx]
    smiles_to_fp_map = {s: fps_arr[i] for i, s in enumerate(valid_smiles)}

    df = df[df["smiles"].isin(smiles_to_fp_map)]
    print(f"  {len(df)} paires avec fingerprint valide")

    # Matrice expression CCLE
    expr_matrix = np.load(ccle_expr).astype(np.float32)
    cell_to_idx = {cid: i for i, cid in enumerate(ccle_id_list)}

    fps_out, exprs_out, aucs_out, mol_ids, cell_ids = [], [], [], [], []

    for _, row in df.iterrows():
        smi    = row["smiles"]
        depmap = row["depmap_id"]
        if depmap not in cell_to_idx:
            continue
        fps_out.append(smiles_to_fp_map[smi])
        exprs_out.append(expr_matrix[cell_to_idx[depmap]])
        aucs_out.append(float(row["auc"]))
        mol_ids.append(smi)
        cell_ids.append(depmap)

    return (np.array(fps_out),
            np.array(exprs_out),
            np.array(aucs_out, dtype=np.float32).reshape(-1, 1),
            mol_ids, cell_ids)


# =============================================================================
#  Cross-validation
# =============================================================================

def make_splits(mol_ids, cell_ids, n_folds, seed):
    """
    Génère 4 types de splits pour chaque fold :
      - mol_seen   : test = molécules vues en train (split aléatoire des paires)
      - mol_unseen : test = molécules entièrement absentes du train
      - cell_seen  : test = lignées vues en train
      - cell_unseen: test = lignées entièrement absentes du train
    """
    mol_ids  = np.array(mol_ids)
    cell_ids = np.array(cell_ids)
    n        = len(mol_ids)
    rng      = np.random.default_rng(seed)

    scenarios = {}

    # ── mol_seen / cell_seen : split aléatoire des paires ────────────────────
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    mol_seen_splits  = list(kf.split(np.arange(n)))
    cell_seen_splits = mol_seen_splits   # même split, nom différent

    scenarios["mol_seen"]  = mol_seen_splits
    scenarios["cell_seen"] = cell_seen_splits

    # ── mol_unseen : les molécules du test n'apparaissent pas en train ────────
    unique_mols = np.unique(mol_ids)
    kf_mol      = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    mol_unseen_splits = []
    for train_mols_idx, test_mols_idx in kf_mol.split(unique_mols):
        train_mols = set(unique_mols[train_mols_idx])
        test_mols  = set(unique_mols[test_mols_idx])
        train_idx  = np.where([m in train_mols for m in mol_ids])[0]
        test_idx   = np.where([m in test_mols  for m in mol_ids])[0]
        mol_unseen_splits.append((train_idx, test_idx))
    scenarios["mol_unseen"] = mol_unseen_splits

    # ── cell_unseen : les lignées du test n'apparaissent pas en train ─────────
    unique_cells = np.unique(cell_ids)
    kf_cell      = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cell_unseen_splits = []
    for train_cells_idx, test_cells_idx in kf_cell.split(unique_cells):
        train_cells = set(unique_cells[train_cells_idx])
        test_cells  = set(unique_cells[test_cells_idx])
        train_idx   = np.where([c in train_cells for c in cell_ids])[0]
        test_idx    = np.where([c in test_cells  for c in cell_ids])[0]
        cell_unseen_splits.append((train_idx, test_idx))
    scenarios["cell_unseen"] = cell_unseen_splits

    return scenarios


# =============================================================================
#  Construction du modèle global
# =============================================================================

def build_model(mol_weights_path, cell_weights_path):
    """Construit et charge les poids des deux VAE, retourne le modèle global."""

    # VAE mol
    vae_mol = VAE(**VAE_MOL_HP, name="vae_mol")
    vae_mol(tf.zeros((1, VAE_MOL_HP["input_dim"])))
    vae_mol.load_weights(mol_weights_path)
    print(f"  VAE mol chargé  ({VAE_MOL_HP['latent_dim']}D)")

    # VAE cell
    vae_cell = VAECell(**VAE_CELL_HP, name="vae_cell")
    vae_cell(tf.zeros((1, VAE_CELL_HP["input_dim"])))
    vae_cell.load_weights(cell_weights_path)
    print(f"  VAE cell chargé ({VAE_CELL_HP['latent_dim']}D)")

    model = DrugResponseModel(
        vae_mol       = vae_mol,
        vae_cell      = vae_cell,
        mlp_layers    = MLP_HP["hidden_layers"],
        mlp_dropout   = MLP_HP["dropout_rate"],
        mlp_activation= MLP_HP["activation"],
        loss_weights  = LOSS_WEIGHTS,
        name          = "drug_response_model",
    )
    model.compile(optimizer=keras.optimizers.Adam(TRAIN["learning_rate"]))
    return model


# =============================================================================
#  Évaluation
# =============================================================================

def evaluate(model, fps, exprs, aucs, batch_size):
    """Retourne mse, pearson_r, spearman_r sur un jeu de données."""
    preds = []
    for i in range(0, len(fps), batch_size):
        batch_fp   = fps[i : i + batch_size]
        batch_expr = exprs[i : i + batch_size]
        pred = model((batch_fp, batch_expr), training=False).numpy().flatten()
        preds.append(pred)
    preds   = np.concatenate(preds)
    targets = aucs.flatten()

    mse = float(np.mean((preds - targets) ** 2))
    if len(np.unique(preds)) > 1:
        pcc, _   = pearsonr(preds, targets)
        spc, _   = spearmanr(preds, targets)
    else:
        pcc = spc = 0.0
    return mse, pcc, spc, preds, targets


# =============================================================================
#  Visualisations résultats
# =============================================================================

def plot_results(all_results, outdir):
    """Graphe de synthèse : MSE / PCC / Spearman par scénario."""
    scenarios = list(all_results.keys())
    metrics   = ["mse", "pearson", "spearman"]
    colors    = {"mse": "#f97316", "pearson": "#00e5ff", "spearman": "#7c3aed"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0d1117")

    for ax, metric in zip(axes, metrics):
        ax.set_facecolor("#161b22")
        means = [np.mean([f[metric] for f in all_results[s]]) for s in scenarios]
        stds  = [np.std( [f[metric] for f in all_results[s]]) for s in scenarios]
        bars  = ax.bar(scenarios, means, yerr=stds, capsize=5,
                       color=colors[metric], alpha=0.85,
                       error_kw={"ecolor": "white", "linewidth": 1.5})
        ax.set_title(metric.upper(), color="white", fontweight="bold")
        ax.set_ylabel(metric, color="white")
        ax.tick_params(colors="white", axis="both")
        ax.set_xticklabels(scenarios, rotation=20, ha="right", color="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(stds) * 0.1,
                    f"{mean:.3f}", ha="center", va="bottom",
                    color="white", fontsize=9)

    plt.suptitle("Cross-validation — Modèle global (Drug Response)",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cv_results.png"), dpi=130,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  → {outdir}/cv_results.png")


def plot_scatter(preds, targets, scenario, fold, outdir):
    """Scatter prédit vs réel pour un fold."""
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.scatter(targets, preds, s=4, alpha=0.4, color="#00e5ff")
    mn, mx = min(targets.min(), preds.min()), max(targets.max(), preds.max())
    ax.plot([mn, mx], [mn, mx], "w--", lw=1, alpha=0.5)
    ax.set_xlabel("AUC réel",   color="white")
    ax.set_ylabel("AUC prédit", color="white")
    ax.set_title(f"{scenario} — fold {fold}", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"scatter_{scenario}_fold{fold}.png"),
                dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


# =============================================================================
#  Pipeline principal
# =============================================================================

def main():
    outdir = PATHS["outdir"]
    os.makedirs(outdir, exist_ok=True)
    np.random.seed(TRAIN["random_seed"])
    tf.random.set_seed(TRAIN["random_seed"])

    # ── Données ───────────────────────────────────────────────────────────────
    fps, exprs, aucs, mol_ids, cell_ids = load_prism_pairs(
        PATHS["prism"], PATHS["ccle_expr"], PATHS["ccle_ids"]
    )
    print(f"\nDataset total : {len(fps)} paires")
    print(f"  Molécules uniques : {len(set(mol_ids))}")
    print(f"  Lignées uniques   : {len(set(cell_ids))}")

    # ── Splits ────────────────────────────────────────────────────────────────
    scenarios = make_splits(mol_ids, cell_ids,
                            TRAIN["n_folds"], TRAIN["random_seed"])

    # ── Cross-validation ──────────────────────────────────────────────────────
    all_results = {s: [] for s in scenarios}

    for scenario, splits in scenarios.items():
        print(f"\n{'='*60}")
        print(f"Scénario : {scenario}")
        print(f"{'='*60}")

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            print(f"\n  Fold {fold_idx + 1}/{TRAIN['n_folds']}")

            # Re-construire le modèle avec poids pré-entraînés à chaque fold
            print("  Chargement des poids pré-entraînés...")
            model = build_model(PATHS["mol_weights"], PATHS["cell_weights"])

            # Données train / test
            fps_tr,   fps_te   = fps[train_idx],   fps[test_idx]
            exprs_tr, exprs_te = exprs[train_idx], exprs[test_idx]
            aucs_tr,  aucs_te  = aucs[train_idx],  aucs[test_idx]

            # Dataset TF
            ds_train = (
                tf.data.Dataset
                .from_tensor_slices(((fps_tr, exprs_tr), aucs_tr))
                .shuffle(len(fps_tr))
                .batch(TRAIN["batch_size"])
                .prefetch(tf.data.AUTOTUNE)
            )

            # Early stopping sur la loss de prédiction
            es = keras.callbacks.EarlyStopping(
                monitor="pred_loss", mode="min", patience=10,
                restore_best_weights=True, verbose=0
            )

            model.fit(ds_train, epochs=TRAIN["epochs"],
                      callbacks=[es], verbose=1)

            # Évaluation
            mse, pcc, spc, preds, targets = evaluate(
                model, fps_te, exprs_te, aucs_te, TRAIN["batch_size"]
            )
            print(f"  Test — MSE: {mse:.4f}  PCC: {pcc:.4f}  Spearman: {spc:.4f}")

            all_results[scenario].append({
                "fold": fold_idx, "mse": mse, "pearson": pcc, "spearman": spc
            })

            # Scatter du premier fold de chaque scénario
            if fold_idx == 0:
                plot_scatter(preds, targets, scenario, fold_idx + 1, outdir)

            # Sauvegarde des poids du meilleur fold (premier seulement)
            if fold_idx == 0:
                model.save_weights(
                    os.path.join(outdir, f"model_{scenario}_fold0.weights.h5")
                )

    # ── Résumé ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RÉSUMÉ CROSS-VALIDATION")
    print(f"{'='*60}")
    rows = []
    for scenario, folds in all_results.items():
        row = {
            "scenario"      : scenario,
            "mse_mean"      : np.mean([f["mse"]      for f in folds]),
            "mse_std"       : np.std( [f["mse"]      for f in folds]),
            "pearson_mean"  : np.mean([f["pearson"]   for f in folds]),
            "pearson_std"   : np.std( [f["pearson"]   for f in folds]),
            "spearman_mean" : np.mean([f["spearman"]  for f in folds]),
            "spearman_std"  : np.std( [f["spearman"]  for f in folds]),
        }
        rows.append(row)
        print(f"  {scenario:15s}  "
              f"MSE {row['mse_mean']:.4f}±{row['mse_std']:.4f}  "
              f"PCC {row['pearson_mean']:.4f}±{row['pearson_std']:.4f}  "
              f"Spm {row['spearman_mean']:.4f}±{row['spearman_std']:.4f}")

    df_res = pd.DataFrame(rows)
    df_res.to_csv(os.path.join(outdir, "cv_results.csv"), index=False)

    plot_results(all_results, outdir)

    print(f"\n✅  Résultats dans : {outdir}/")


if __name__ == "__main__":
    main()