import os
os.environ["KERAS_BACKEND"] = "tensorflow"   # comme dans le notebook de référence

import keras
from vae import VAE


# =============================================================================
#  VAE moléculaire
#  Entrée : fingerprint RDKit Daylight-like — vecteur binaire 2048 dims
# =============================================================================

# ── Hyperparamètres ───────────────────────────────────────────────────────────

INPUT_DIM    = 2048   # taille du fingerprint RDKit (fixé par rdkit, ne pas changer)
LATENT_DIM   = 64     # dimension de l'espace latent z_mol

# Architecture de l'encodeur : liste des couches cachées
# [1024, 512, 256] → 3 couches avant les têtes µ et log_var
ENCODER_LAYERS = [1024, 512, 256]

# Décodeur : miroir de l'encodeur si None
# None → [256, 512, 1024] automatiquement
DECODER_LAYERS = None

ACTIVATION       = "relu"
DROPOUT_RATE     = 0.2
USE_BATCH_NORM   = True

# sigmoid car le fingerprint est binaire (0/1) — même logique que le
# notebook qui utilise sigmoid pour reconstruire les pixels MNIST (0-1)
OUTPUT_ACTIVATION = "sigmoid"

# Poids initial du terme KL — faible au départ pour le KL annealing
# On l'augmente progressivement via le callback KLAnnealingCallback (voir plus bas)
LAMBDA_KL_INIT = 1e-4

LEARNING_RATE = 1e-3
BATCH_SIZE    = 256
EPOCHS        = 50


# ── Création du VAE moléculaire ───────────────────────────────────────────────

vae_mol = VAE(
    input_dim         = INPUT_DIM,
    latent_dim        = LATENT_DIM,
    encoder_layers    = ENCODER_LAYERS,
    decoder_layers    = DECODER_LAYERS,
    activation        = ACTIVATION,
    dropout_rate      = DROPOUT_RATE,
    use_batch_norm    = USE_BATCH_NORM,
    output_activation = OUTPUT_ACTIVATION,
    lambda_kl         = LAMBDA_KL_INIT,
    name              = "vae_moleculaire"
)

vae_mol.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))


# ── Affichage du résumé ───────────────────────────────────────────────────────

vae_mol.summary_full()


class KLAnnealingCallback(keras.callbacks.Callback):
    """Augmente lambda_kl linéairement de 0 à lambda_kl_max
    sur les premières kl_warmup_epochs époques.

    Inspiré de la pratique standard pour les VAE sur données biologiques.
    """

    def __init__(self, lambda_kl_max=1e-3, kl_warmup_epochs=10):
        super().__init__()
        self.lambda_kl_max     = lambda_kl_max
        self.kl_warmup_epochs  = kl_warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        # Augmentation linéaire : 0 → lambda_kl_max sur kl_warmup_epochs
        new_lambda = min(
            self.lambda_kl_max,
            self.lambda_kl_max * (epoch + 1) / self.kl_warmup_epochs
        )
        self.model.lambda_kl = new_lambda
        print(f"\n[KL Annealing] Époque {epoch+1} — lambda_kl = {new_lambda:.2e}")


#── Entraînement (à décommenter quand les données sont prêtes) ────────────────

import numpy as np

# Charger les fingerprints depuis data/processed/
# Shape attendue : (n_molecules, 2048) — valeurs 0.0 ou 1.0
fingerprints = np.load("data/processed/fingerprints_train.npy").astype("float32")

kl_callback = KLAnnealingCallback(lambda_kl_max=1e-3, kl_warmup_epochs=10)

history = vae_mol.fit(
    fingerprints,
    epochs     = EPOCHS,
    batch_size = BATCH_SIZE,
    callbacks  = [kl_callback],
)

# Sauvegarder les poids
vae_mol.save_weights("results/vae_mol_weights.weights.h5")

# Obtenir les représentations latentes
z_mean, z_log_var, z_mol = vae_mol.encode(fingerprints)
