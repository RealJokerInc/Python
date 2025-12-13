"""
Test script to verify UNet segmentation setup works correctly.
"""
import os
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from scipy.ndimage import convolve

print("=" * 60)
print("Testing UNet Segmentation Setup")
print("=" * 60)

# Define functions
def process_image(im: Image.Image, target_size=(512, 512)) -> np.ndarray:
    """Process image: grayscale + Sobel edge channel for UNet input."""
    im_gray = im.convert('L')
    im_gray = im_gray.resize(target_size)
    arr_gray = np.asarray(im_gray, dtype=np.float32) / 255.0

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    grad_x = convolve(arr_gray, sobel_x)
    grad_y = convolve(arr_gray, sobel_y)
    sobel_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    sobel_magnitude = sobel_magnitude / (sobel_magnitude.max() + 1e-8)

    arr_combined = np.stack([arr_gray, sobel_magnitude], axis=-1)
    return arr_combined

def process_mask(mask: Image.Image, target_size=(512, 512)) -> np.ndarray:
    """Process segmentation mask: resize and normalize to 0-1."""
    mask_gray = mask.convert('L')
    mask_gray = mask_gray.resize(target_size)
    mask_array = np.asarray(mask_gray, dtype=np.float32) / 255.0
    mask_array = np.expand_dims(mask_array, axis=-1)
    return mask_array

def double_conv_block(x, n_filters):
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    return x

def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)
    return f, p

def upsample_block(x, conv_features, n_filters):
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = layers.concatenate([x, conv_features])
    x = layers.Dropout(0.3)(x)
    x = double_conv_block(x, n_filters)
    return x

def build_unet_model(input_shape=(512, 512, 2)):
    inputs = layers.Input(shape=input_shape)
    f1, p1 = downsample_block(inputs, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)
    bottleneck = double_conv_block(p4, 1024)
    u6 = upsample_block(bottleneck, f4, 512)
    u7 = upsample_block(u6, f3, 256)
    u8 = upsample_block(u7, f2, 128)
    u9 = upsample_block(u8, f1, 64)
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(u9)
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    return unet_model

# Test 1: Load sample data
print("\n1. Testing data loading...")
dataset_path = Path('/Users/lemon/.cache/kagglehub/datasets/jeetblahiri/bccd-dataset-with-mask/versions/1')
train_img_dir = dataset_path / "BCCD Dataset with mask" / "train" / "original"
train_mask_dir = dataset_path / "BCCD Dataset with mask" / "train" / "mask"

if not train_img_dir.exists():
    print(f"ERROR: Dataset not found at {train_img_dir}")
    print("Please ensure the dataset is downloaded.")
    exit(1)

sample_files = list(train_img_dir.glob("*.png"))[:2]
print(f"Found {len(sample_files)} sample images for testing")

# Test 2: Process images
print("\n2. Testing image processing...")
X_samples = []
y_samples = []

for img_path in sample_files:
    mask_path = train_mask_dir / img_path.name
    with Image.open(img_path) as img, Image.open(mask_path) as mask:
        X_samples.append(process_image(img))
        y_samples.append(process_mask(mask))

X_samples = np.array(X_samples)
y_samples = np.array(y_samples)

print(f"X_samples shape: {X_samples.shape} (expected: (2, 512, 512, 2))")
print(f"y_samples shape: {y_samples.shape} (expected: (2, 512, 512, 1))")
print(f"X value range: [{X_samples.min():.3f}, {X_samples.max():.3f}]")
print(f"y value range: [{y_samples.min():.3f}, {y_samples.max():.3f}]")

assert X_samples.shape == (2, 512, 512, 2), "X_samples shape mismatch!"
assert y_samples.shape == (2, 512, 512, 1), "y_samples shape mismatch!"
print("✓ Image processing works correctly!")

# Test 3: Build model
print("\n3. Testing UNet model construction...")
model = build_unet_model(input_shape=(512, 512, 2))
print(f"Model built successfully!")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

assert model.input_shape == (None, 512, 512, 2), "Input shape mismatch!"
assert model.output_shape == (None, 512, 512, 1), "Output shape mismatch!"
print("✓ UNet architecture is correct!")

# Test 4: Compile model
print("\n4. Testing model compilation...")
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
print("✓ Model compiled successfully!")

# Test 5: Forward pass
print("\n5. Testing forward pass...")
predictions = model.predict(X_samples, verbose=0)
print(f"Predictions shape: {predictions.shape} (expected: (2, 512, 512, 1))")
print(f"Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")

assert predictions.shape == (2, 512, 512, 1), "Predictions shape mismatch!"
assert 0 <= predictions.min() <= predictions.max() <= 1, "Predictions out of [0, 1] range!"
print("✓ Forward pass works correctly!")

# Test 6: Training step
print("\n6. Testing training step (1 batch)...")
history = model.fit(
    X_samples, y_samples,
    epochs=1,
    batch_size=2,
    verbose=1
)
print(f"Training loss: {history.history['loss'][0]:.4f}")
print(f"Training accuracy: {history.history['accuracy'][0]:.4f}")
print("✓ Training step works!")

print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✓")
print("=" * 60)
print("\nThe UNet segmentation setup is working correctly.")
print("You can now run the full notebook to train on the complete dataset.")
