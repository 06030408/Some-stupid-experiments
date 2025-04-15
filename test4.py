import tensorflow as tf
import numpy as np
import datetime
import os

# Disable GPU (optional)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 1. Configuration parameters
dataset_path = "./stl10_binary/stl10_binary"  # Modify to your actual path
image_size = 96
channels = 3
batch_size = 32


# 2. Manual STL-10 data loading
def load_stl_binary(file_path):
    """Load STL-10 binary file"""
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
        # Reshape to NHWC format [samples, height, width, channels]
        data = data.reshape((-1, channels, image_size, image_size))  # [N, C, H, W]
        data = np.transpose(data, (0, 2, 3, 1))  # Convert to [N, H, W, C]
        return data.astype(np.float32) / 255.0


# Load data
train_images = load_stl_binary(os.path.join(dataset_path, "train_X.bin"))[:1000]  # Take first 1000 samples
test_images = load_stl_binary(os.path.join(dataset_path, "test_X.bin"))[:800]  # Take first 800 samples

# 3. Create datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_images))
train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_images))
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


# 4. Define model (keep unchanged)
def create_model():
    inputs = tf.keras.Input(shape=(image_size, image_size, channels))

    # Encoder
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)  # 48x48
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D()(x)  # 24x24
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    encoded = tf.keras.layers.MaxPooling2D()(x)  # 12x12

    # Decoder
    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same')(encoded)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
    outputs = tf.keras.layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


# 5. Custom callback (keep unchanged)
class ImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data, log_dir):
        super().__init__()
        self.test_data = test_data
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        test_samples = self.test_data[:4]
        generated = self.model.predict(test_samples, verbose=0)
        with self.writer.as_default():
            generated_images = tf.clip_by_value(generated * 255, 0, 255).numpy().astype(np.uint8)
            tf.summary.image("Generated Images", generated_images, step=epoch, max_outputs=4)


# 6. Training configuration
learning_rates = [0.01, 0.001, 0.0001]
epochs = 20
test_losses = []

for lr in learning_rates:
    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mse',
                  metrics=['mae'])

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"_lr{lr}"
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        ),
        ImageCallback(test_images, log_dir)
    ]

    print(f"\nTraining with learning rate: {lr}")
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        callbacks=callbacks
    )

    test_loss, _ = model.evaluate(test_dataset, verbose=0)
    test_losses.append(test_loss)

# 7. Results output
print("\nTest Losses (MSE):")
for lr, loss in zip(learning_rates, test_losses):
    print(f"Learning rate {lr}: {loss:.4f}")

# 8. Usage instructions
print("\nTensorBoard viewing steps:")
print("1. Execute in terminal: tensorboard --logdir logs/")
print("2. Open browser to: http://localhost:6006")
print("3. Check training metrics under 'Scalars' tab")
print("4. View model structure under 'Graphs' tab")
print("5. Inspect generated samples under 'Images' tab")
