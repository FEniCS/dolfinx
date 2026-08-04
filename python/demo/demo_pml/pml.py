"""PML coordinate stretching and transformed material tensors."""

import ufl


def pml_coordinates(x, alpha: float, k0: complex, l_dom: float, l_pml: float):
    return x + 1j * alpha / k0 * x * (ufl.algebra.Abs(x) - l_dom / 2) / (l_pml / 2 - l_dom / 2) ** 2


def create_eps_mu(pml, eps_bkg, mu_bkg):
    J = ufl.grad(pml)

    # Transform the 2x2 Jacobian into a 3x3 matrix.
    J = ufl.as_matrix(((J[0, 0], 0, 0), (0, J[1, 1], 0), (0, 0, 1)))

    A = ufl.inv(J)
    eps_pml = ufl.det(J) * A * eps_bkg * ufl.transpose(A)
    mu_pml = ufl.det(J) * A * mu_bkg * ufl.transpose(A)
    return eps_pml, mu_pml


def create_pml_tensors(domain, mesh_params, problem_params):
    """New complex coordinates
    xy_pml : PML corners
    x_pml : PML rectabgles along x
    y_pml PML rectangles along y
    
    The PML region can be interpreted as a material having, in general, anisotropic,
    inhomogeneous and complex permittivity
    """
    x = ufl.SpatialCoordinate(domain)
    alpha = problem_params.alpha
    k0 = problem_params.k0
    l_dom = mesh_params.l_dom
    l_pml = mesh_params.l_pml

    # PML corners: r' = (x', y')
    xy_pml = ufl.as_vector(
        (
            pml_coordinates(x[0], alpha, k0, l_dom, l_pml),
            pml_coordinates(x[1], alpha, k0, l_dom, l_pml),
        )
    )

    # PML rectangles along x: r' = (x', y)
    x_pml = ufl.as_vector((pml_coordinates(x[0], alpha, k0, l_dom, l_pml), x[1]))

    # PML rectangles along y: r' = (x, y')
    y_pml = ufl.as_vector((x[0], pml_coordinates(x[1], alpha, k0, l_dom, l_pml)))

    eps_x, mu_x = create_eps_mu(x_pml, problem_params.eps_bkg, 1)
    eps_y, mu_y = create_eps_mu(y_pml, problem_params.eps_bkg, 1)
    eps_xy, mu_xy = create_eps_mu(xy_pml, problem_params.eps_bkg, 1)

    return eps_x, mu_x, eps_y, mu_y, eps_xy, mu_xy
