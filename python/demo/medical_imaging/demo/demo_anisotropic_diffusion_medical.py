import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from petsc4py import PETSc
import ufl
from pathlib import Path
import argparse
from PIL import Image
from skimage.metrics import structural_similarity
from scipy import ndimage
from scipy.interpolate import griddata
from dolfinx.fem.petsc import assemble_matrix, assemble_vector


def load_image(filepath, normalize=True, target_size=None):    
    img = Image.open(filepath).convert('L')
    
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
    
    domain = mesh.create_rectangle(
        comm,
        [[0.0, 0.0], [float(width), float(height)]],
        [width-1, height-1],
        cell_type=mesh.CellType.triangle
    )
    
    return domain


def image_to_function(image, V):
    u = fem.Function(V)
    height, width = image.shape
    
    def image_values(x):
        values = np.zeros(x.shape[1])
        
        for i in range(x.shape[1]):
            x_coord = x[0, i]
            y_coord = x[1, i]
            
            px = int(np.clip(np.round(x_coord), 0, width - 1))
            py = int(np.clip(np.round(height - 1 - y_coord), 0, height - 1))
            
            values[i] = image[py, px]
        
        return values
    
    u.interpolate(image_values)
    return u


def function_to_image(u, image_shape):
    height, width = image_shape
    
    mesh_obj = u.function_space.mesh
    coords = mesh_obj.geometry.x[:, :2]
    values = u.x.array
    
    x_pixels = np.linspace(0, width - 1, width)
    y_pixels = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x_pixels, y_pixels)
    
    Y_flipped = height - 1 - Y
    
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
    return structural_similarity(
        original, denoised, 
        data_range=denoised.max() - denoised.min()
    )


def edge_preservation_index(original, denoised):
    sobel_orig = ndimage.sobel(original)
    sobel_denoised = ndimage.sobel(denoised)
    
    sobel_orig = sobel_orig / (np.max(sobel_orig) + 1e-10)
    sobel_denoised = sobel_denoised / (np.max(sobel_denoised) + 1e-10)
    
    numerator = np.sum(sobel_orig * sobel_denoised)
    denominator = np.sqrt(np.sum(sobel_orig**2) * np.sum(sobel_denoised**2))
    
    epi = numerator / (denominator + 1e-10)
    return epi


def create_synthetic_test_image(output_file="data/synthetic_test.png", size=(512, 512)):
    img = np.zeros(size, dtype=np.float64)
    y, x = np.ogrid[:size[0], :size[1]]
    
    cx1, cy1, r1 = size[1]//4, size[0]//4, 60
    mask1 = (x - cx1)**2 + (y - cy1)**2 <= r1**2
    img[mask1] = 1.0
    
    cx2, cy2, r2 = 3*size[1]//4, 3*size[0]//4, 80
    mask2 = (x - cx2)**2 + (y - cy2)**2 <= r2**2
    img[mask2] = 0.9
    
    img[size[0]//3:2*size[0]//3, size[1]//2-40:size[1]//2+40] = 0.8
    
    np.random.seed(42)
    noise = np.random.normal(0, 0.03, size)
    noisy_img = np.clip(img + noise, 0, 1)
    
    img_uint8 = (noisy_img * 255).astype(np.uint8)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_uint8).save(output_file, quality=100, optimize=False)
    
    return output_file


class AnisotropicDiffusion:
    
    def __init__(self, image_shape, kappa=20.0, dt=0.05, comm=MPI.COMM_WORLD):
        self.image_shape = image_shape
        self.kappa = kappa
        self.dt = dt
        self.comm = comm

        self.domain = create_image_mesh(image_shape, comm)
        self.V = fem.functionspace(self.domain, ("Lagrange", 1))
        
        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)
        self.u_n = fem.Function(self.V)
        
    def diffusion_coefficient(self, grad_u):
        grad_magnitude = ufl.sqrt(ufl.dot(grad_u, grad_u) + 1e-10)
        c = 1.0 / (1.0 + (grad_magnitude / self.kappa)**2)
        return c
    
    def setup_variational_problem(self):
        c = self.diffusion_coefficient(ufl.grad(self.u_n))
        
        F = (self.u - self.u_n) * self.v * ufl.dx + \
            self.dt * c * ufl.dot(ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx
        
        a = ufl.lhs(F)
        L = ufl.rhs(F)
        
        return a, L
    
    def solve_timestep(self, a, L):
        A = assemble_matrix(fem.form(a))
        A.assemble()
        b = assemble_vector(fem.form(L))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        solver = PETSc.KSP().create(self.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.CG)
        
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.JACOBI)
        
        u_new = fem.Function(self.V)
        solver.solve(b, u_new.x.petsc_vec)
        u_new.x.scatter_forward()
        
        return u_new
    
    def denoise(self, image, n_iterations=30, verbose=True):
        self.u_n = image_to_function(image, self.V)
        
        a, L = self.setup_variational_problem()
        
        for iteration in range(n_iterations):
            if verbose and self.comm.rank == 0:
                print(f"Iteration {iteration + 1}/{n_iterations}")
            
            u_new = self.solve_timestep(a, L)
            self.u_n.x.array[:] = u_new.x.array[:]
            a, L = self.setup_variational_problem()
        
        denoised = function_to_image(self.u_n, self.image_shape)
        return denoised


def main():
    parser = argparse.ArgumentParser(
        description="Anisotropic diffusion for medical image denoising"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image (uses synthetic if not provided)"
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=20.0,
        help="Edge threshold parameter (default: 20.0)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Number of iterations (default: 30)"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.05,
        help="Time step size (default: 0.05)"
    )
    
    args = parser.parse_args()
    
    if args.image:
        print(f"Demo: Anisotropic Diffusion on Medical Image\n")
        print(f"Loading image: {args.image}")
        image = load_image(args.image, normalize=True, target_size=(512, 512))
    else:
        print(f"Demo: Anisotropic Diffusion on Synthetic Image\n")
        test_image_path = "data/synthetic_test.png"
        
        if not Path(test_image_path).exists():
            print("Creating synthetic test image...")
            create_synthetic_test_image(test_image_path)
        
        print(f"Loading image: {test_image_path}")
        image = load_image(test_image_path, normalize=True)
    
    print(f"Image shape: {image.shape}\n")
    
    print("Initializing anisotropic diffusion solver...")
    solver = AnisotropicDiffusion(
        image_shape=image.shape,
        kappa=args.kappa,
        dt=args.dt
    )
    
    print(f"\nApplying anisotropic diffusion...")
    print(f"Parameters: kappa={args.kappa}, dt={args.dt}, iterations={args.iterations}\n")
    denoised = solver.denoise(image, n_iterations=args.iterations, verbose=True)
    
    print(f"\nQuality Metrics:\n")
    psnr = compute_psnr(image, denoised)
    ssim = compute_ssim(image, denoised)
    epi = edge_preservation_index(image, denoised)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    print(f"Edge Preservation Index: {epi:.4f}")
    
    Path("output").mkdir(exist_ok=True)
    output_path = "output/denoised.png"
    save_image(denoised, output_path)
    print(f"\nDenoised image saved to: {output_path}")


if __name__ == "__main__":
    main()
