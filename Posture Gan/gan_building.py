import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import layers, Model

# Define the encoder (LSTM-based)
def build_encoder(input_shape, latent_dim):
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.LSTM(64, return_sequences=False)(x)
    latent = layers.Dense(latent_dim, activation='relu', name='encoder_output')(x)
    return Model(inputs, latent, name='encoder')

# Define the generator (LSTM-based)
def build_generator(latent_dim, output_shape):
    inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.RepeatVector(output_shape[0])(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(output_shape[1], activation='tanh'))(x)  # Output between -1 and 1
    return Model(inputs, outputs, name='generator')

# Define the discriminator (LSTM-based)
def build_discriminator(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.LSTM(64, return_sequences=False)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name='discriminator')

# Model hyperparameters
sequence_length = 10  # Number of time steps per sequence
input_dim = 6  # Glasses_Position + Rotation (6 features per time step)
latent_dim = 5  # Encoded posture representation

encoder = build_encoder((sequence_length, input_dim), latent_dim)
generator = build_generator(latent_dim, (sequence_length, input_dim))
discriminator = build_discriminator((sequence_length, input_dim))

discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

discriminator.trainable = False

gan_input = layers.Input(shape=(latent_dim,))
generated_data = generator(gan_input)
discriminator_output = discriminator(generated_data)
gan = Model(gan_input, discriminator_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

print("✅ LSTM-based Sequence GAN Model Created")
print("Encoder Summary:")
encoder.summary()
print("Generator Summary:")
generator.summary()
print("Discriminator Summary:")
discriminator.summary()
