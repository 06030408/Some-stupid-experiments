import tensorflow as tf
import numpy as np
import datetime
import os

# Disable GPU (optional)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 1. Create image dataset using CIFAR10
(image_size, channels) = (32, 3)  # CIFAR10 dimensions

# Load CIFAR10 dataset
(train_images, _), (test_images, _) = tf.keras.datasets.cifar10.load_data()

# Take 1/5 of training data and 1/10 of test data
train_images = train_images[:10000]  # Original 50000, take first 10000 (1/5)
test_images = test_images[:1000]     # Original 10000, take first 1000 (1/10)
# Normalize to [0,1] and convert to float32
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# Create TensorFlow Dataset
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_images))
train_dataset = train_dataset.shuffle(50000).batch(batch_size)  # Use full dataset size for shuffling
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_images))
test_dataset = test_dataset.batch(batch_size)


# 2. Define image generation model (Autoencoder for 32x32 input)
def create_model():
    inputs = tf.keras.Input(shape=(image_size, image_size, channels))

    # Encoder
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)  # 16x16
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D()(x)  # 8x8

    # Decoder
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)  # 16x16
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)  # 32x32
    outputs = tf.keras.layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


# Custom callback to save generated images (remain unchanged)
class ImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data, log_dir):
        super().__init__()
        self.test_data = test_data
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        test_samples = self.test_data[:4]
        generated = self.model.predict(test_samples)
        with self.writer.as_default():
            generated_images = tf.clip_by_value(generated * 255, 0, 255).numpy().astype(np.uint8)
            tf.summary.image("Generated Images", generated_images, step=epoch, max_outputs=4)


# 3. Training configuration (hyperparameters remain unchanged)
learning_rates = [0.1, 0.01, 0.001]
epochs = 10
test_losses = []

for lr in learning_rates:
    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mse',
                  metrics=['mae'])

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"_lr{lr}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )
    image_callback = ImageCallback(test_images, log_dir)

    print(f"\nTraining with learning rate: {lr}")
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        callbacks=[tensorboard_callback, image_callback]
    )

    test_loss, test_mae = model.evaluate(test_dataset)
    test_losses.append(test_loss)

# 4. Output results (remain unchanged)
print("\nTest Losses (MSE):")
for lr, loss in zip(learning_rates, test_losses):
    print(f"Learning rate {lr}: {loss:.4f}")

# 5. Usage instructions (remain unchanged)
print("\nTo view TensorBoard results:")
print("1. In terminal, run: tensorboard --logdir logs/")
print("2. Open http://localhost:6006 in your browser")
print("3. View training metrics in 'Scalars' tab")
print("4. View model architecture in 'Graphs' tab")
print("5. View generated images in 'Images' tab")
