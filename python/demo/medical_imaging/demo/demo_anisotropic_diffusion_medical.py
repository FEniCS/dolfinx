import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, LinearProblem
import ufl
from ufl import grad, div, dx, dot, sqrt, inner
from petsc4py import PETSc
import sys
import os
from basix.ufl import element

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.image_processing import (
    load_image, save_image, create_image_mesh, 
    image_to_function, function_to_image,
    compute_psnr, compute_ssim, edge_preservation_index
)
from data.download_cbis_ddsm import create_synthetic_test_image

class AnisotropicDiffusion:
    def __init__(self, image_shape, kappa=30.0, dt=0.1, comm=MPI.COMM_WORLD):
        self.image_shape = image_shape
        self.kappa = kappa
        self.dt = dt
        self.comm = comm

        self.domain = create_image_mesh(image_shape, comm)
        
        P1 = element("Lagrange", self.domain.basix_cell(), 1)
        self.V = fem.functionspace(self.domain, P1)
        
        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)
        
        self.u_n = fem.Function(self.V)
        
    def diffusion_coefficient(self, grad_u):
        # Compute gradient magnitude with small epsilon to avoid division by zero
        grad_magnitude = sqrt(dot(grad_u, grad_u) + 1e-10)
        
        # Perona-Malik diffusion coefficient
        c = 1.0 / (1.0 + (grad_magnitude / self.kappa)**2)
        
        return c
    
    def setup_variational_problem(self):
        c = self.diffusion_coefficient(grad(self.u_n))
        
        F = (self.u - self.u_n) * self.v * dx + \
            self.dt * c * dot(grad(self.u), grad(self.v)) * dx
        
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
    
    def denoise(self, image, n_iterations=20, verbose=True):
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

def demo_synthetic_image():
    print(f"Demo: Anisotropic Diffusion on Synthetic Image\n")
    
    # Load or create synthetic test image
    test_image_path = "data/synthetic_test.png"
    
    if not os.path.exists(test_image_path):
        print("Creating synthetic test image...")
        os.makedirs("data", exist_ok=True)
        create_synthetic_test_image(test_image_path)
    
    # Load image
    print(f"Loading image: {test_image_path}")
    image = load_image(test_image_path, normalize=True)
    print(f"Image shape: {image.shape}")
    
    solver = AnisotropicDiffusion(
        image_shape=image.shape,
        kappa=20.0,
        dt=0.05
    )
    
    print("\nApplying anisotropic diffusion...")
    denoised = solver.denoise(image, n_iterations=40, verbose=True)
    
    # Compute quality metrics
    print(f"Quality Metrics:\n")
    psnr = compute_psnr(image, denoised)
    ssim = compute_ssim(image, denoised)
    epi = edge_preservation_index(image, denoised)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    print(f"Edge Preservation Index: {epi:.4f}")
    
    # Save results
    os.makedirs("output", exist_ok=True)
    save_image(denoised, "output/synthetic_denoised.png")
    print(f"\nDenoised image saved to: output/synthetic_denoised.png")
    
    return image, denoised


def demo_medical_image(image_path):
    print(f"Demo: Anisotropic Diffusion on Medical Image\n")
    
    # Load image
    print(f"Loading image: {image_path}")
    
    target_size = (512, 512)
    image = load_image(image_path, normalize=True, target_size=target_size)
    print(f"Image shape: {image.shape}")
    
    print("\nInitializing anisotropic diffusion solver...")
    solver = AnisotropicDiffusion(
        image_shape=image.shape,
        kappa=20.0,
        dt=0.05
    )
    
    print("\nApplying anisotropic diffusion...")
    denoised = solver.denoise(image, n_iterations=30, verbose=True)
    
    print(f"Quality Metrics:\n")
    psnr = compute_psnr(image, denoised)
    ssim = compute_ssim(image, denoised)
    epi = edge_preservation_index(image, denoised)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    print(f"Edge Preservation Index: {epi:.4f}")
    
    # Save results
    os.makedirs("output", exist_ok=True)
    output_path = "output/medical_denoised.png"
    save_image(denoised, output_path)
    print(f"\nDenoised image saved to: {output_path}")
    
    return image, denoised


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Anisotropic diffusion for medical image denoising"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image (if not provided, uses synthetic test image)"
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=30.0,
        help="Gradient threshold (default: 30.0)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of iterations (default: 20)"
    )
    
    args = parser.parse_args()
    
    if args.image:
        demo_medical_image(args.image)
    else:
        demo_synthetic_image()