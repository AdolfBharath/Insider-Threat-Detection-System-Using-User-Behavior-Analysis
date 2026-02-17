from __future__ import annotations

from typing import Any


def build_lstm_autoencoder(seq_len: int, feature_dim: int, latent_dim: int = 32, dropout: float = 0.2):
    # Single deep learning model: LSTM Autoencoder
    try:
        import tensorflow as tf
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "TensorFlow is required for LSTM Autoencoder. Install with: pip install tensorflow"
        ) from e

    inputs = tf.keras.Input(shape=(seq_len, feature_dim), name="seq")

    x = tf.keras.layers.LSTM(64, return_sequences=True, name="enc_lstm1")(inputs)
    x = tf.keras.layers.Dropout(dropout, name="enc_do1")(x)
    z = tf.keras.layers.LSTM(latent_dim, return_sequences=False, name="enc_lstm2")(x)

    x = tf.keras.layers.RepeatVector(seq_len, name="rep")(z)
    x = tf.keras.layers.LSTM(latent_dim, return_sequences=True, name="dec_lstm1")(x)
    x = tf.keras.layers.Dropout(dropout, name="dec_do1")(x)
    x = tf.keras.layers.LSTM(64, return_sequences=True, name="dec_lstm2")(x)

    outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(feature_dim), name="recon")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model
