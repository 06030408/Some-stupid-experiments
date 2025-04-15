import tensorflow as tf
import numpy as np
import datetime
import os

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 1. Create simulated dataset
num_samples = 100
image_size = 128

# Generate random images and masks (2 classes)
images = np.random.rand(num_samples, image_size, image_size, 3).astype(np.float32)
masks = np.zeros((num_samples, image_size, image_size), dtype=np.int32)

# Create simple mask pattern (central square)
for i in range(num_samples):
    size = np.random.randint(30, 50)
    x = np.random.randint(20, image_size - size - 20)
    y = np.random.randint(20, image_size - size - 20)
    masks[i, x:x + size, y:y + size] = 1

# Split into train and test sets
train_images, test_images = images[:80], images[80:]
train_masks, test_masks = masks[:80], masks[80:]

# Create TensorFlow Dataset
batch_size = 8
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
train_dataset = train_dataset.shuffle(80).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_masks))
test_dataset = test_dataset.batch(batch_size)


# 2. Define semantic segmentation model
def create_model():
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))

    # Encoder
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)  # 64x64

    # Decoder
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D()(x)  # 128x128

    # Output layer
    outputs = tf.keras.layers.Conv2D(2, 1, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


# 3. Training configuration
learning_rates = [0.1, 0.01, 0.001]
epochs = 10
test_accuracies = []

for lr in learning_rates:
    # Create new model
    model = create_model()

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Configure TensorBoard
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"_lr{lr}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )

    # Train model
    print(f"\nTraining with learning rate: {lr}")
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        callbacks=[tensorboard_callback]
    )

    # Evaluate model
    test_loss, test_acc = model.evaluate(test_dataset)
    test_accuracies.append(test_acc)

# 4. Output results
print("\nTest Accuracies:")
for lr, acc in zip(learning_rates, test_accuracies):
    print(f"Learning rate {lr}: {acc:.4f}")

# 5. Usage instructions
print("\nTo view TensorBoard results:")
print("1. In terminal, run: tensorboard --logdir logs/")
print("2. Open http://localhost:6006 in your browser")
print("3. View training curves in 'Scalars' tab and model architecture in 'Graphs' tab")
