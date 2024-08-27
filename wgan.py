from IPython.display import clear_output

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# WGAN Training
class WGAN:
    def __init__(self, generator, critic, noise_dim, critic_extra_steps=5, gp_weight=10.0, c_lr=1e-3, g_lr=1e-3): # Try values between 15 and 25
        self.generator = generator
        self.critic = critic
        self.noise_dim = noise_dim
        self.critic_extra_steps = critic_extra_steps
        self.gp_weight = gp_weight
        self.stop = False
        self.stop = False
        self.continue_flag = False
        self.current_epoch = 0
        self.epochs = 200

        # self.g_optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9) # ORIGINAL
        # self.c_optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9) # ORIGINAL
        self.c_optimizer = tf.keras.optimizers.Adam(keras.optimizers.schedules.ExponentialDecay(
                                                    initial_learning_rate=c_lr,
                                                    decay_steps=1000,
                                                    decay_rate=0.9
                                                    ), beta_1=0.5, beta_2=0.9)
        # self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        self.g_optimizer = tf.keras.optimizers.Adam(keras.optimizers.schedules.ExponentialDecay(
                                                    initial_learning_rate=g_lr,
                                                    decay_steps=1000,
                                                    decay_rate=0.9
                                                    ), beta_1=0.5, beta_2=0.9)
    def gradient_penalty(self, real, fake):
        alpha = tf.random.uniform([real.shape[0], 1, 1], 0.0, 1.0, dtype=tf.float32)
        diff = fake - real
        interpolated = real + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @tf.function
    def train_step(self, real_data):
        real_data = tf.cast(real_data, tf.float32)  # Ensure real_data is float32
        batch_size = tf.shape(real_data)[0]
        for _ in range(self.critic_extra_steps):
            noise = tf.random.normal([batch_size, self.noise_dim], dtype=tf.float32)
            with tf.GradientTape() as tape:
                fake_data = self.generator(noise, training=True)
                critic_real = self.critic(real_data, training=True)
                critic_fake = self.critic(fake_data, training=True)
                critic_loss = tf.reduce_mean(critic_fake) - tf.reduce_mean(critic_real)
                gp = self.gradient_penalty(real_data, fake_data)
                critic_loss += self.gp_weight * gp
            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        noise = tf.random.normal([batch_size, self.noise_dim], dtype=tf.float32)
        with tf.GradientTape() as tape:
            fake_data = self.generator(noise, training=True)
            critic_fake = self.critic(fake_data, training=True)
            generator_loss = -tf.reduce_mean(critic_fake)
        generator_grad = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(generator_grad, self.generator.trainable_variables))

        return {"critic_loss": critic_loss, "generator_loss": generator_loss}
    
    def stop_training(self):
        self.stop = True
    
    def continue_training(self):
        self.continue_flag = True
        self.stop = False
        self.epochs += 200  # Add more epochs
        print(f"Training will continue. Epochs now set to {self.epochs}")

    # Plotting function
    def plot_loss(self, G_loss, D_loss, epochs):
        plt.figure(figsize=(6, 4))
        plt.title(f'Development of Training Losses During Training. Epochs: {epochs}')
        plt.plot(D_loss, label='Critic Loss')
        plt.plot(G_loss, label='Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def train(self, dataset, epochs):
        self.epochs = epochs
        G_loss = []
        D_loss = []
        
        while self.current_epoch < self.epochs:
            if self.stop:
                print(f"Training stopped at epoch {self.current_epoch}")
                break

            G_list = []
            D_list = []

            for batch in dataset:
                losses = self.train_step(batch)
                G_list.append(losses['generator_loss'].numpy())
                D_list.append(losses['critic_loss'].numpy())

            # Calculate mean loss for the epoch
            G_loss.append(np.mean(G_list))
            D_loss.append(np.mean(D_list))
            
            # Produce plots for the losses during training
            clear_output(wait=True)
            self.plot_loss(G_loss, D_loss, self.epochs)
            
            print(f'Epoch {self.current_epoch + 1}, Critic Loss: {D_loss[-1]:.4f}, Generator Loss: {G_loss[-1]:.4f}')
            
            self.current_epoch += 1

            if self.continue_flag:
                self.continue_flag = False
                print("Continuing training...")

        return G_loss, D_loss

class ImprovedWGAN(WGAN):
    def __init__(self, generator, critic, noise_dim, critic_extra_steps=5, gp_weight=10.0, c_lr=1e-4, g_lr=1e-4, feature_matching_weight=0.1):
        super().__init__(generator, critic, noise_dim, critic_extra_steps, gp_weight, c_lr, g_lr)
        self.feature_matching_weight = feature_matching_weight

    def gradient_penalty(self, real, fake):
        alpha = tf.random.uniform([real.shape[0], 1, 1], 0.0, 1.0)
        diff = fake - real
        interpolated = real + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred, _ = self.critic(interpolated, training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @tf.function
    def train_step(self, real_data):
        real_data = tf.cast(real_data, tf.float32)
        batch_size = tf.shape(real_data)[0]
        
        for _ in range(self.critic_extra_steps):
            noise = tf.random.normal([batch_size, self.noise_dim])
            with tf.GradientTape() as tape:
                fake_data = self.generator(noise, training=True)
                critic_real, _ = self.critic(real_data, training=True)
                critic_fake, _ = self.critic(fake_data, training=True)
                critic_loss = tf.reduce_mean(critic_fake) - tf.reduce_mean(critic_real)
                gp = self.gradient_penalty(real_data, fake_data)
                critic_loss += self.gp_weight * gp
            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        noise = tf.random.normal([batch_size, self.noise_dim])
        with tf.GradientTape() as tape:
            fake_data = self.generator(noise, training=True)
            critic_fake, fake_features = self.critic(fake_data, training=True)
            generator_loss = -tf.reduce_mean(critic_fake)
            
            # Feature matching loss
            _, real_features = self.critic(real_data, training=False)
            feature_matching_loss = tf.reduce_mean(tf.abs(tf.reduce_mean(real_features, axis=1) - tf.reduce_mean(fake_features, axis=1)))
            
            generator_loss += self.feature_matching_weight * feature_matching_loss
            
            # Add L2 regularization to encourage smoother outputs
            l2_loss = tf.reduce_mean(tf.square(fake_data[:, 1:] - fake_data[:, :-1]))
            generator_loss += 0.1 * l2_loss

        generator_grad = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(generator_grad, self.generator.trainable_variables))

        return {"critic_loss": critic_loss, "generator_loss": generator_loss}

    def train(self, dataset, epochs):
        G_loss = []
        D_loss = []
        
        for epoch in range(epochs):
            G_list = []
            D_list = []

            for batch in dataset:
                losses = self.train_step(batch)
                G_list.append(losses['generator_loss'].numpy())
                D_list.append(losses['critic_loss'].numpy())

            G_loss.append(np.mean(G_list))
            D_loss.append(np.mean(D_list))
            clear_output(wait=True)
            self.plot_loss(G_loss, D_loss)

            if epoch % 10 == 0:
                # self.plot_loss(G_loss, D_loss)
                print(f'Epoch {epoch + 1}, Critic Loss: {D_loss[-1]:.4f}, Generator Loss: {G_loss[-1]:.4f}')

        return G_loss, D_loss
