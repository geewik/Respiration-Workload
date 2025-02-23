import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import os
import pandas as pd


# Load preprocessed dataset
def load_data(data_path):
    subjects = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    data_list = []

    for subject in subjects:
        file_path = os.path.join(data_path, subject, "processed_dataset.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Subject'] = subject  # Tag subject ID
            data_list.append(df)

    return pd.concat(data_list, ignore_index=True)


processed_data = load_data("D:/aria-marker/gan_preprocessed")

# Extract different posture data
looking_up = processed_data[processed_data['Posture'] == 'Looking Up']
looking_down = processed_data[processed_data['Posture'] == 'Looking Down']
looking_straight = processed_data[processed_data['Posture'] == 'Straight']

# Convert to numpy arrays for training
looking_up_data = looking_up.drop(columns=['Posture', 'Subject']).values
looking_down_data = looking_down.drop(columns=['Posture', 'Subject']).values
looking_straight_data = looking_straight.drop(columns=['Posture', 'Subject']).values

# Define training parameters
epochs = 500
batch_size = 128
noise_dim = 5  # Latent space dimension
sequence_length = 10
input_dim = 6

# Import models from gan_building.py
from gan_building import build_encoder, build_generator, build_discriminator

# Initialize models
encoder_up = build_encoder((sequence_length, input_dim), noise_dim)
encoder_down = build_encoder((sequence_length, input_dim), noise_dim)
encoder_straight = build_encoder((sequence_length, input_dim), noise_dim)
generator = build_generator(noise_dim, (sequence_length, input_dim))
discriminator = build_discriminator((sequence_length, input_dim))

discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy',
                      metrics=['accuracy'])
discriminator.trainable = False

gan_up = tf.keras.Model(encoder_up.input, discriminator(generator(encoder_up.output)))
gan_down = tf.keras.Model(encoder_down.input, discriminator(generator(encoder_down.output)))
gan_straight = tf.keras.Model(encoder_straight.input, discriminator(generator(encoder_straight.output)))

gan_up.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')
gan_down.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')
gan_straight.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')

# Training loop
for epoch in range(epochs):
    for _ in range(len(looking_up_data) // batch_size):
        idx_up = np.random.randint(0, looking_up_data.shape[0], batch_size)
        idx_down = np.random.randint(0, looking_down_data.shape[0], batch_size)
        idx_straight = np.random.randint(0, looking_straight_data.shape[0], batch_size)

        real_data_up = looking_up_data[idx_up].reshape(batch_size, sequence_length, input_dim)
        real_data_down = looking_down_data[idx_down].reshape(batch_size, sequence_length, input_dim)
        real_data_straight = looking_straight_data[idx_straight].reshape(batch_size, sequence_length, input_dim)

        encoded_up = encoder_up(real_data_up)
        encoded_down = encoder_down(real_data_down)
        encoded_straight = encoder_straight(real_data_straight)

        noise_up = tf.random.normal([batch_size, noise_dim])
        noise_down = tf.random.normal([batch_size, noise_dim])
        noise_straight = tf.random.normal([batch_size, noise_dim])

        generated_up = generator(noise_up)
        generated_down = generator(noise_down)
        generated_straight = generator(noise_straight)

        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))

        discriminator.trainable = True
        d_loss_real_up = discriminator.train_on_batch(real_data_up, real_labels)
        d_loss_fake_up = discriminator.train_on_batch(generated_up, fake_labels)
        d_loss_real_down = discriminator.train_on_batch(real_data_down, real_labels)
        d_loss_fake_down = discriminator.train_on_batch(generated_down, fake_labels)
        d_loss_real_straight = discriminator.train_on_batch(real_data_straight, real_labels)
        d_loss_fake_straight = discriminator.train_on_batch(generated_straight, fake_labels)

        discriminator.trainable = False
        g_loss_up = gan_up.train_on_batch(real_data_up, real_labels)
        g_loss_down = gan_down.train_on_batch(real_data_down, real_labels)
        g_loss_straight = gan_straight.train_on_batch(real_data_straight, real_labels)

    print(
        f'Epoch {epoch + 1}/{epochs} - Generator Loss (Up/Down/Straight): {g_loss_up:.4f}, {g_loss_down:.4f}, {g_loss_straight:.4f}')

# Save models
os.makedirs('D:/aria-marker/encoders', exist_ok=True)
encoder_up.save('D:/aria-marker/encoders/encoder_up.keras')
encoder_down.save('D:/aria-marker/encoders/encoder_down.keras')
encoder_straight.save('D:/aria-marker/encoders/encoder_straight.keras')
generator.save('D:/aria-marker/encoders/generator.keras')
discriminator.save('D:/aria-marker/encoders/discriminator.keras')

print("✅ Training completed! Encoders and models saved.")
