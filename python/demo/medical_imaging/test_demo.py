import numpy as np
import pytest
from mpi4py import MPI
from pathlib import Path
import sys
import ufl
from dolfinx import fem

sys.path.insert(0, str(Path(__file__).parent / "demo"))

from demo_anisotropic_diffusion_medical import (
    AnisotropicDiffusion, 
    create_synthetic_test_image, 
    create_image_mesh, 
    load_image, 
    compute_psnr, 
    compute_ssim, 
    edge_preservation_index
)


def test_synthetic_image_generation():
    output_file = "test_synthetic.png"
    try:
        result = create_synthetic_test_image(output_file, size=(128, 128))
        assert Path(output_file).exists()
        assert result == output_file
    finally:
        if Path(output_file).exists():
            Path(output_file).unlink()


def test_image_loading():
    output_file = "test_load.png"
    try:
        create_synthetic_test_image(output_file, size=(64, 64))
        image = load_image(output_file, normalize=True)
        
        assert image.shape == (64, 64)
        assert image.dtype == np.float64
        assert 0.0 <= image.min() <= 1.0
        assert 0.0 <= image.max() <= 1.0
    finally:
        if Path(output_file).exists():
            Path(output_file).unlink()


def test_anisotropic_diffusion_basic():
    output_file = "test_diffusion.png"
    try:
        create_synthetic_test_image(output_file, size=(32, 32))
        image = load_image(output_file, normalize=True)
        
        solver = AnisotropicDiffusion(
            image_shape=image.shape,
            kappa=20.0,
            dt=0.05,
            comm=MPI.COMM_WORLD
        )
        
        denoised = solver.denoise(image, n_iterations=2, verbose=False)
        
        assert denoised.shape == image.shape
        assert np.isfinite(denoised).all()
        assert denoised.min() >= 0.0
        assert denoised.max() <= 1.5
        
    finally:
        if Path(output_file).exists():
            Path(output_file).unlink()


def test_quality_metrics():
    original = np.random.rand(32, 32)
    denoised = original + 0.01 * np.random.randn(32, 32)
    denoised = np.clip(denoised, 0, 1)
    
    psnr = compute_psnr(original, denoised)
    assert np.isfinite(psnr)
    assert psnr > 0
    
    ssim = compute_ssim(original, denoised)
    assert np.isfinite(ssim)
    assert 0 <= ssim <= 1
    
    epi = edge_preservation_index(original, denoised)
    assert np.isfinite(epi)
    assert 0 <= epi <= 1


def test_mesh_creation():
    image_shape = (32, 32)
    domain = create_image_mesh(image_shape, comm=MPI.COMM_WORLD)
    
    assert domain is not None
    assert domain.topology.dim == 2


def test_diffusion_coefficient():
    solver = AnisotropicDiffusion(
        image_shape=(32, 32),
        kappa=20.0,
        dt=0.05
    )
    
    u = fem.Function(solver.V)
    grad_u = ufl.grad(u)
    
    c = solver.diffusion_coefficient(grad_u)
    
    assert c is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])