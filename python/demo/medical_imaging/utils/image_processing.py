import numpy as np
from PIL import Image
from dolfinx import mesh, fem
from mpi4py import MPI
import ufl
from scipy import ndimage
from skimage.metrics import structural_similarity
from scipy.interpolate import griddata
from basix.ufl import element


def load_image(filepath, normalize=True, target_size=None):
    img = Image.open(filepath).convert('L')  # Convert to grayscale
    
    if target_size is not None:
        img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
    
    image = np.array(img, dtype=np.float64)
    
    if normalize:
        image = image / 255.0
    
    return image


def save_image(image, filepath, denormalize=True):
    if denormalize:
        image = np.clip(image * 255.0, 0, 255)
    
    img = Image.fromarray(image.astype(np.uint8))
    img.save(filepath)


def create_image_mesh(image_shape, comm=MPI.COMM_WORLD):
    height, width = image_shape
    
    # Create unit square mesh and scale to image dimensions
    domain = mesh.create_rectangle(
        comm,
        [[0.0, 0.0], [float(width), float(height)]],
        [width-1, height-1],
        cell_type=mesh.CellType.triangle
    )
    
    return domain


def image_to_function(image, V):
    u = fem.Function(V)
    
    # Get mesh coordinates
    height, width = image.shape
    
    # Interpolate image values onto function space
    def image_values(x):
        values = np.zeros(x.shape[1])
        
        for i in range(x.shape[1]):
            x_coord = x[0, i]
            y_coord = x[1, i]
            
            # Convert to pixel coordinates (with bounds checking)
            px = int(np.clip(np.round(x_coord), 0, width - 1))
            py = int(np.clip(np.round(height - 1 - y_coord), 0, height - 1))
            
            values[i] = image[py, px]
        
        return values
    
    u.interpolate(image_values)
    
    return u


def function_to_image(u, image_shape): 
    height, width = image_shape
    
    # Get mesh coordinates and function values
    mesh = u.function_space.mesh
    coords = mesh.geometry.x[:, :2]  # Get x, y coordinates (drop z)
    values = u.x.array
    
    x_pixels = np.linspace(0, width - 1, width)
    y_pixels = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x_pixels, y_pixels)
    
    Y_flipped = height - 1 - Y
    
    # Interpolate function values onto regular pixel grid
    # Use 'linear' interpolation for smooth results
    image = griddata(
        coords,
        values,
        (X, Y_flipped),
        method='linear',
        fill_value=0.0
    )
    
    image = np.nan_to_num(image, nan=0.0)
    
    return image


def compute_psnr(original, denoised, max_value=1.0):
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * np.log10(max_value / np.sqrt(mse))
    return psnr


def compute_ssim(original, denoised):   
    return structural_similarity(original, denoised, data_range=denoised.max() - denoised.min())


def edge_preservation_index(original, denoised):
    sobel_orig = ndimage.sobel(original)
    sobel_denoised = ndimage.sobel(denoised)
    
    # Normalize
    sobel_orig = sobel_orig / (np.max(sobel_orig) + 1e-10)
    sobel_denoised = sobel_denoised / (np.max(sobel_denoised) + 1e-10)
    
    # Compute correlation
    numerator = np.sum(sobel_orig * sobel_denoised)
    denominator = np.sqrt(np.sum(sobel_orig**2) * np.sum(sobel_denoised**2))
    
    epi = numerator / (denominator + 1e-10)
    
    return epi


def add_gaussian_noise(image, var=0.01, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.normal(0, np.sqrt(var), image.shape)
    noisy = image + noise
    
    return np.clip(noisy, 0, 1)