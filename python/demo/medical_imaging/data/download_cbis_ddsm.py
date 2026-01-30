import os
import numpy as np
import zipfile
from PIL import Image
from pathlib import Path
import argparse
import shutil

def download_dataset(output_dir="data/cbis_ddsm"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    os.system(f"kaggle datasets download -d awsaf49/cbis-ddsm-breast-cancer-image-dataset -p {output_dir}")
    
    zip_file = output_path / "cbis-ddsm-breast-cancer-image-dataset.zip"
    if zip_file.exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        zip_file.unlink()
        return True
    else:
        print("Error: Dataset download failed.")
        return False

def prepare_sample_images(dataset_dir="data/cbis_ddsm", sample_dir="data/samples", n_samples=5):
    sample_path = Path(sample_dir)
    sample_path.mkdir(parents=True, exist_ok=True)
    
    dataset_path = Path(dataset_dir)
    image_files = list(dataset_path.rglob("*.png"))[:n_samples]
    
    if not image_files:
        print("No images found. Please download the dataset first.")
        return
    
    print(f"Copying {len(image_files)} sample images...")
    
    for i, img_file in enumerate(image_files):
        img = Image.open(img_file).convert('L')
        max_size = 1024
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        output_file = sample_path / f"sample_{i:02d}.png"
        img.save(output_file)
        print(f"  Saved: {output_file}")
    
    print(f"Sample images saved to {sample_dir}/")

def create_synthetic_test_image(output_file="data/synthetic_test.png", size=(512, 512)):
    # Create clean image with geometric shapes
    img = np.zeros(size, dtype=np.float64)
    
    y, x = np.ogrid[:size[0], :size[1]]
    
    # Circle 1 (top-left) - bright white
    cx1, cy1, r1 = size[1]//4, size[0]//4, 60
    mask1 = (x - cx1)**2 + (y - cy1)**2 <= r1**2
    img[mask1] = 1.0
    
    # Circle 2 (bottom-right) - slightly dimmer
    cx2, cy2, r2 = 3*size[1]//4, 3*size[0]//4, 80
    mask2 = (x - cx2)**2 + (y - cy2)**2 <= r2**2
    img[mask2] = 0.9
    
    # Rectangle (center)
    img[size[0]//3:2*size[0]//3, size[1]//2-40:size[1]//2+40] = 0.8
    
    # Add light Gaussian noise
    np.random.seed(42)  # Reproducible
    noise = np.random.normal(0, 0.03, size)
    noisy_img = img + noise
    
    # Clip to [0, 1]
    noisy_img = np.clip(noisy_img, 0, 1)
    
    # Convert to 8-bit with high quality
    img_uint8 = (noisy_img * 255).astype(np.uint8)
    
    # Save with maximum quality
    Image.fromarray(img_uint8).save(output_file, quality=100, optimize=False)
    
    print(f"Improved synthetic test image saved to {output_file}")
    print(f"    Image size: {size}")
    print(f"    Normalized range: [0, 1]")
    print(f"    Noise level: LIGHT (std=0.03)")
    print(f"    Features: 2 circles + 1 rectangle")
    print(f"    Quality: HIGH (no compression)")
    
    return output_file

if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description="Download and prepare CBIS-DDSM dataset")
    parser.add_argument("--download", action="store_true", help="Download full dataset")
    parser.add_argument("--samples", action="store_true", help="Create sample subset")
    parser.add_argument("--synthetic", action="store_true", help="Create synthetic test image")
    parser.add_argument("--all", action="store_true", help="Do all of the above")
    
    args = parser.parse_args()
    
    if args.all or args.synthetic:
        create_synthetic_test_image()
    
    if args.all or args.download:
        success = download_dataset()
        if success and (args.all or args.samples):
            prepare_sample_images()
    elif args.samples:
        prepare_sample_images()
    
    if not any([args.download, args.samples, args.synthetic, args.all]):
        parser.print_help()