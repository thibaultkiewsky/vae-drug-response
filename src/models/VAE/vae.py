import numpy as np
import tensorflow as tf
import keras
from keras import layers


# =============================================================================
#  Sampling layer
# =============================================================================

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z.
    z = z_mean + exp(0.5 * z_log_var) * epsilon,   epsilon ~ N(0, I)
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim   = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# =============================================================================
#  Classe VAE générique
# =============================================================================

class VAE(keras.Model):
    """VAE générique avec encodeur et décodeur MLP configurables.

    Paramètres
    ----------
    input_dim : int
        Dimension du vecteur d'entrée (ex: 2048 pour fingerprint, 8087 pour gènes).

    latent_dim : int
        Dimension de l'espace latent z (ex: 64).

    encoder_layers : list[int]
        Taille de chaque couche cachée de l'encodeur.
        Ex: [512, 256] → deux couches cachées avant les têtes µ et log_var.

    decoder_layers : list[int]
        Taille de chaque couche cachée du décodeur.
        Ex: [256, 512] → deux couches cachées avant la reconstruction.
        Si None, on utilise l'inverse de encoder_layers automatiquement.

    activation : str
        Fonction d'activation des couches cachées (défaut : "relu").

    dropout_rate : float
        Taux de dropout appliqué après chaque couche cachée (0 = désactivé).

    use_batch_norm : bool
        Active la BatchNormalization après chaque couche cachée.

    output_activation : str
        Activation de la couche de sortie du décodeur.
        "sigmoid" pour des entrées binaires (fingerprints),
        "linear"  pour des données continues (expression génique).

    lambda_kl : float
        Poids du terme KL dans la loss totale. Permet le KL annealing
        en passant une valeur croissante au fil de l'entraînement.

    name : str
        Nom du modèle (affiché dans les logs et summary).
    """

    def __init__(
        self,
        input_dim,
        latent_dim      = 64,
        encoder_layers  = [512, 256],
        decoder_layers  = None,
        activation      = "relu",
        dropout_rate    = 0.2,
        use_batch_norm  = True,
        output_activation = "sigmoid",
        lambda_kl       = 1e-3,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.input_dim         = input_dim
        self.latent_dim        = latent_dim
        self.lambda_kl         = lambda_kl
        self.output_activation = output_activation

        # Si decoder_layers non précisé → miroir de l'encodeur
        if decoder_layers is None:
            decoder_layers = list(reversed(encoder_layers))

        # ── Construit l'encodeur et le décodeur ──────────────────────────────
        self.encoder = self._build_encoder(
            input_dim, latent_dim, encoder_layers,
            activation, dropout_rate, use_batch_norm
        )
        self.decoder = self._build_decoder(
            latent_dim, input_dim, decoder_layers,
            activation, dropout_rate, use_batch_norm,
            output_activation
        )

        # ── Trackers de métriques ────────────────────────────────────────────
        self.total_loss_tracker         = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker            = keras.metrics.Mean(name="kl_loss")

    # -------------------------------------------------------------------------
    #  Construction de l'encodeur
    # -------------------------------------------------------------------------

    def _build_encoder(self, input_dim, latent_dim, hidden_layers,
                        activation, dropout_rate, use_batch_norm):
        """Construit le modèle Keras de l'encodeur.

        Architecture :
            Input(input_dim)
            → [Dense → (BatchNorm) → Activation → (Dropout)] x N couches
            → z_mean     : Dense(latent_dim)   # pas d'activation
            → z_log_var  : Dense(latent_dim)   # pas d'activation
            → z          : Sampling()([z_mean, z_log_var])
        """
        encoder_inputs = keras.Input(shape=(input_dim,), name="encoder_input")
        x = encoder_inputs

        for i, units in enumerate(hidden_layers):
            x = layers.Dense(units, name=f"enc_dense_{i}")(x)
            if use_batch_norm:
                x = layers.BatchNormalization(name=f"enc_bn_{i}")(x)
            x = layers.Activation(activation, name=f"enc_act_{i}")(x)
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate, name=f"enc_drop_{i}")(x)

        # Deux têtes sans activation 
        z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

        # Reparametrization trick via la couche Sampling
        z = Sampling(name="z")([z_mean, z_log_var])

        return keras.Model(
            encoder_inputs,
            [z_mean, z_log_var, z],
            name="encoder"
        )

    # -------------------------------------------------------------------------
    #  Construction du décodeur
    # -------------------------------------------------------------------------

    def _build_decoder(self, latent_dim, output_dim, hidden_layers,
                        activation, dropout_rate, use_batch_norm,
                        output_activation):
        """Construit le modèle Keras du décodeur.

        Architecture :
            Input(latent_dim)
            → [Dense → (BatchNorm) → Activation → (Dropout)] x N couches
            → Dense(output_dim, activation=output_activation)
        """
        latent_inputs = keras.Input(shape=(latent_dim,), name="decoder_input")
        x = latent_inputs

        for i, units in enumerate(hidden_layers):
            x = layers.Dense(units, name=f"dec_dense_{i}")(x)
            if use_batch_norm:
                x = layers.BatchNormalization(name=f"dec_bn_{i}")(x)
            x = layers.Activation(activation, name=f"dec_act_{i}")(x)
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate, name=f"dec_drop_{i}")(x)

        # Couche de sortie — sigmoid pour binaire, linear pour continu
        decoder_outputs = layers.Dense(
            output_dim,
            activation=output_activation,
            name="decoder_output"
        )(x)

        return keras.Model(latent_inputs, decoder_outputs, name="decoder")

    # -------------------------------------------------------------------------
    #  Métriques exposées 
    # -------------------------------------------------------------------------

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    # -------------------------------------------------------------------------
    #  train_step 
    #  On y ajoute : poids lambda_kl pour le KL annealing
    # -------------------------------------------------------------------------

    def train_step(self, data):
        """Boucle d'entraînement sur un batch.

        Ajout par rapport au notebook : pondération lambda_kl sur le terme KL,
        ce qui permet de faire du KL annealing en modifiant self.lambda_kl
        à chaque époque via un callback.

        Loss totale :
            L = L_reconstruction + lambda_kl * L_KL
        """
        with tf.GradientTape() as tape:

            # ── Passe avant ──────────────────────────────────────────────────
            z_mean, z_log_var, z = self.encoder(data, training=True)
            reconstruction = self.decoder(z, training=True)

            # ── Loss de reconstruction ───────────────────────────────────────
            # BCE pour entrées binaires (fingerprints)
            # Remplacer par MSE pour l'encodeur génomique (voir vae_cell.py)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=0
                )
            )

            # ── Loss KL — formule identique au notebook de référence ─────────
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # ── Loss totale avec pondération KL ──────────────────────────────
            total_loss = reconstruction_loss + self.lambda_kl * kl_loss

        # ── Rétropropagation ─────────────────────────────────────────────────
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # ── Mise à jour des métriques ────────────────────────────────────────
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss":               self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss":            self.kl_loss_tracker.result(),
        }

    # -------------------------------------------------------------------------
    #  call — encode puis décode (utile pour tester le modèle)
    # -------------------------------------------------------------------------

    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        reconstruction = self.decoder(z, training=training)
        return reconstruction

    # -------------------------------------------------------------------------
    #  Méthodes utilitaires
    # -------------------------------------------------------------------------

    def encode(self, x):
        """Retourne (z_mean, z_log_var, z) pour un batch d'entrées."""
        return self.encoder(x, training=False)

    def decode(self, z):
        """Reconstruit l'entrée depuis un vecteur latent z."""
        return self.decoder(z, training=False)

    def summary_full(self):
        """Affiche le summary de l'encodeur et du décodeur séparément."""
        print("=" * 60)
        print("ENCODEUR")
        print("=" * 60)
        self.encoder.summary()
        print("\n" + "=" * 60)
        print("DÉCODEUR")
        print("=" * 60)
        self.decoder.summary()
