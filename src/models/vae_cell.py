import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
from keras import layers
from vae import VAE, Sampling


# =============================================================================
#  VAE génomique
#  Entrée : profil d'expression génique — 8087 gènes (CCLE, après feature selection)
#  Différence clé avec vae_mol.py : loss de reconstruction = MSE (pas BCE)
#  car les valeurs d'expression sont continues (log2 fold-change), pas binaires
# =============================================================================

# ── Hyperparamètres ───────────────────────────────────────────────────────────

INPUT_DIM  = 8087   # gènes retenus après feature selection (PCC > 0.2 / ≥10 molécules)
LATENT_DIM = 64     # dimension de l'espace latent z_cell

# Encodeur plus large car l'entrée est plus grande qu'un fingerprint
ENCODER_LAYERS = [2048, 1024, 256]

# Décodeur : miroir automatique → [256, 1024, 2048]
DECODER_LAYERS = None

ACTIVATION     = "relu"
DROPOUT_RATE   = 0.2
USE_BATCH_NORM = True

# linear car l'expression génique est continue (log2 fold-change ∈ ℝ)
# ≠ vae_mol qui utilise sigmoid pour des valeurs binaires
OUTPUT_ACTIVATION = "linear"

LAMBDA_KL_INIT = 1e-4
LEARNING_RATE  = 1e-3
BATCH_SIZE     = 128    # plus petit que vae_mol car moins de lignées (~476) que de molécules
EPOCHS         = 50


# =============================================================================
#  Sous-classe VAECell
#  On hérite de VAE et on surcharge uniquement train_step pour utiliser
#  la MSE comme loss de reconstruction (au lieu de la BCE du notebook)
# =============================================================================

class VAECell(VAE):
    """VAE génomique — identique à VAE sauf la loss de reconstruction.

    Le notebook de référence utilise la BCE car MNIST est binaire (pixels 0-1).
    Ici l'expression génique est continue → on remplace par la MSE.
    Tout le reste (Sampling, KL, train_step structure) est identique.
    """

    def train_step(self, data):
        with tf.GradientTape() as tape:

            # ── Passe avant — identique au notebook ──────────────────────────
            z_mean, z_log_var, z = self.encoder(data, training=True)
            reconstruction = self.decoder(z, training=True)

            # ── Loss de reconstruction : MSE (≠ BCE du notebook) ─────────────
            # MSE adaptée aux données continues (expression log2 fold-change)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(data - reconstruction),   # (batch, 8087)
                    axis=1
                )
            )

            # ── Loss KL — formule identique au notebook de référence ─────────
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # ── Loss totale ───────────────────────────────────────────────────
            total_loss = reconstruction_loss + self.lambda_kl * kl_loss

        # ── Rétropropagation — identique au notebook ─────────────────────────
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


# ── Création du VAE génomique ─────────────────────────────────────────────────

vae_cell = VAECell(
    input_dim         = INPUT_DIM,
    latent_dim        = LATENT_DIM,
    encoder_layers    = ENCODER_LAYERS,
    decoder_layers    = DECODER_LAYERS,
    activation        = ACTIVATION,
    dropout_rate      = DROPOUT_RATE,
    use_batch_norm    = USE_BATCH_NORM,
    output_activation = OUTPUT_ACTIVATION,
    lambda_kl         = LAMBDA_KL_INIT,
    name              = "vae_genomique"
)

vae_cell.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))


# ── Affichage du résumé ───────────────────────────────────────────────────────

vae_cell.summary_full()


# =============================================================================
#  Callback KL Annealing — même logique que vae_mol.py
# =============================================================================

class KLAnnealingCallback(keras.callbacks.Callback):
    def __init__(self, lambda_kl_max=1e-3, kl_warmup_epochs=10):
        super().__init__()
        self.lambda_kl_max    = lambda_kl_max
        self.kl_warmup_epochs = kl_warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        new_lambda = min(
            self.lambda_kl_max,
            self.lambda_kl_max * (epoch + 1) / self.kl_warmup_epochs
        )
        self.model.lambda_kl = new_lambda
        print(f"\n[KL Annealing] Époque {epoch+1} — lambda_kl = {new_lambda:.2e}")


# ── Entraînement (à décommenter quand les données sont prêtes) ────────────────

# import numpy as np
#
# # Charger les profils d'expression depuis data/processed/
# # Shape attendue : (n_cell_lines, 8087) — valeurs log2 fold-change
# gene_expr = np.load("data/processed/gene_expression_train.npy").astype("float32")
#
# kl_callback = KLAnnealingCallback(lambda_kl_max=1e-3, kl_warmup_epochs=10)
#
# history = vae_cell.fit(
#     gene_expr,
#     epochs     = EPOCHS,
#     batch_size = BATCH_SIZE,
#     callbacks  = [kl_callback],
# )
#
# # Sauvegarder les poids
# vae_cell.save_weights("results/vae_cell_weights.weights.h5")
#
# # Obtenir les représentations latentes
# z_mean, z_log_var, z_cell = vae_cell.encode(gene_expr)
