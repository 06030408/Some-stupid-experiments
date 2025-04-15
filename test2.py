import tensorflow as tf
import numpy as np
import datetime
import os

# Disable GPU (optional)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 1. Create image dataset
num_samples = 100
image_size = 128

# Generate random image data (input and target are the same for autoencoder)
images = np.random.rand(num_samples, image_size, image_size, 3).astype(np.float32)

# Split into train and test sets
train_images, test_images = images[:80], images[80:]

# Create TensorFlow Dataset
batch_size = 8
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_images))
train_dataset = train_dataset.shuffle(80).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_images))
test_dataset = test_dataset.batch(batch_size)


# 2. Define image generation model (autoencoder)
def create_model():
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))

    # Encoder
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)  # 64x64
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D()(x)  # 32x32

    # Decoder
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)  # 64x64
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)  # 128x128
    outputs = tf.keras.layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


# Custom callback for saving generated images
class ImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data, log_dir):
        super().__init__()
        self.test_data = test_data
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        # Generate test images
        test_samples = self.test_data[:4]
        generated = self.model.predict(test_samples)

        # Write images to TensorBoard
        with self.writer.as_default():
            # Convert to uint8 and scale values
            generated_images = tf.clip_by_value(generated * 255, 0, 255).numpy().astype(np.uint8)
            tf.summary.image("Generated Images", generated_images, step=epoch, max_outputs=4)


# 3. Training configuration
learning_rates = [0.1, 0.01, 0.001]
epochs = 10
test_losses = []

for lr in learning_rates:
    # Create new model
    model = create_model()

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mse',
                  metrics=['mae'])

    # Configure TensorBoard
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"_lr{lr}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )

    # Image generation callback
    image_callback = ImageCallback(test_images, log_dir)

    # Train model
    print(f"\nTraining with learning rate: {lr}")
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        callbacks=[tensorboard_callback, image_callback]
    )

    # Evaluate model
    test_loss, test_mae = model.evaluate(test_dataset)
    test_losses.append(test_loss)

# 4. Output results
print("\nTest Losses (MSE):")
for lr, loss in zip(learning_rates, test_losses):
    print(f"Learning rate {lr}: {loss:.4f}")

# 5. Usage instructions
print("\nTo view TensorBoard results:")
print("1. In terminal, run: tensorboard --logdir logs/")
print("2. Open http://localhost:6006 in your browser")
print("3. View training metrics in 'Scalars' tab")
print("4. View model architecture in 'Graphs' tab")
print("5. View generated images in 'Images' tab")
