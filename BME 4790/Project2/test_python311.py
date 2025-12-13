"""
Test script to verify compatibility with Python 3.11.5
"""
import sys
import os
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import convolve

print("=" * 60)
print("Python 3.11 Compatibility Test")
print("=" * 60)

# Check Python version
print(f"\n1. Python Version Check:")
print(f"   Python: {sys.version}")
print(f"   Version info: {sys.version_info}")

if sys.version_info < (3, 11):
    print(f"   WARNING: Running Python {sys.version_info.major}.{sys.version_info.minor}, not 3.11+")
else:
    print(f"   ✓ Python 3.11+ detected")

# Check required packages
print(f"\n2. Package Versions:")
try:
    import numpy as np
    print(f"   NumPy: {np.__version__}")
except ImportError as e:
    print(f"   NumPy: NOT INSTALLED - {e}")

try:
    from PIL import Image
    print(f"   Pillow: {Image.__version__}")
except ImportError as e:
    print(f"   Pillow: NOT INSTALLED - {e}")

try:
    import scipy
    print(f"   SciPy: {scipy.__version__}")
except ImportError as e:
    print(f"   SciPy: NOT INSTALLED - {e}")

try:
    import matplotlib
    print(f"   Matplotlib: {matplotlib.__version__}")
except ImportError as e:
    print(f"   Matplotlib: NOT INSTALLED - {e}")

try:
    import sklearn
    print(f"   scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"   scikit-learn: NOT INSTALLED - {e}")

try:
    import cv2
    print(f"   OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"   OpenCV: NOT INSTALLED - {e}")

# Test dataset access
print(f"\n3. Dataset Location Test:")
dataset_path = Path("./BCCD Dataset with mask")
if dataset_path.exists():
    print(f"   ✓ Dataset found at: {dataset_path.absolute()}")

    train_path = dataset_path / "train" / "original"
    if train_path.exists():
        num_images = len(list(train_path.glob("*.png")))
        print(f"   ✓ Training images: {num_images}")

    test_path = dataset_path / "test" / "original"
    if test_path.exists():
        num_images = len(list(test_path.glob("*.png")))
        print(f"   ✓ Test images: {num_images}")
else:
    print(f"   ERROR: Dataset not found at {dataset_path.absolute()}")
    print(f"   Please ensure the dataset is in the project folder")

# Test data processing functions
print(f"\n4. Data Processing Test:")

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

try:
    train_img_dir = dataset_path / "train" / "original"
    train_mask_dir = dataset_path / "train" / "mask"

    if train_img_dir.exists():
        sample_files = list(train_img_dir.glob("*.png"))[:2]

        if sample_files:
            X_samples = []
            y_samples = []

            for img_path in sample_files:
                mask_path = train_mask_dir / img_path.name
                with Image.open(img_path) as img, Image.open(mask_path) as mask:
                    X_samples.append(process_image(img))
                    y_samples.append(process_mask(mask))

            X_samples = np.array(X_samples)
            y_samples = np.array(y_samples)

            print(f"   ✓ Processed {len(X_samples)} samples")
            print(f"   ✓ X shape: {X_samples.shape}")
            print(f"   ✓ y shape: {y_samples.shape}")

            assert X_samples.shape == (2, 512, 512, 2), "X shape mismatch!"
            assert y_samples.shape == (2, 512, 512, 1), "y shape mismatch!"
            print(f"   ✓ Data processing works correctly!")
        else:
            print(f"   ERROR: No sample images found")
    else:
        print(f"   ERROR: Train directory not found")

except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Python 3.11 Compatibility Check Complete")
print("=" * 60)
