"""Electromagnetic scattering from a wire with PML
Copyright (C) 2022 Michele Castriotta, Igor Baratta, Jørgen S. Dokken

This demo illustrates how to:
 - Use complex quantities in FEniCSx
 - Setup and solve Maxwell's equations
 - Implement (rectangular) perfectly matched layers (PMLs)

Run from this directory with:
    python main.py
or, in parallel, for example:
    mpirun -n 2 python main.py
"""
# First, we import the required modules
from mpi4py import MPI

from config import MeshParameters, PhysicalConstants, ProblemParameters, Tags
from mesh_wire import create_dolfinx_mesh
from visualize import load_pyvista, make_output_folder, save_field, visualize_mesh
from postprocess import compute_efficiencies, print_and_check_efficiencies
from solve import (
    check_complex_mode,
    create_function_space,
    create_measures,
    create_permittivity,
    create_total_field,
    create_variational_forms,
    interpolate_background_field,
    solve_scattered_field,
)


def main():
    
    check_complex_mode()
    
    #initialize parameters needed
    constants = PhysicalConstants()
    tags = Tags()
    mesh_params = MeshParameters()
    problem_params = ProblemParameters()

    #generate mesh
    mesh_data = create_dolfinx_mesh(mesh_params, tags, comm=MPI.COMM_WORLD)

    out_folder = make_output_folder("output_pml")
    pyvista = load_pyvista()
    visualize_mesh(mesh_data, out_folder, pyvista)

    # element to represent the electric field:
    V = create_function_space(mesh_data.mesh, problem_params.degree)
    
    # interpolate $\mathbf{E}_b$ into the function space $V$,
    Eb = interpolate_background_field(V, problem_params)
    
    # Define a space function for the permittivity D
    # that takes the value D for cells inside the wire, while
    # it takes the value of the background permittivity in
    # the background region:
    D, eps = create_permittivity(mesh_data, tags, problem_params)

    # Solve the weak form in DOLFINx.
    a, L, dx = create_variational_forms(mesh_data, tags, mesh_params, problem_params, V, Eb, eps)
    Esh = solve_scattered_field(a, L, mesh_data.mesh)

    save_field(
        mesh_data.mesh,
        V,
        Esh,
        problem_params.degree,
        out_folder,
        "Esh.bp",
        pyvista=pyvista,
        image_name="Esh.png",
    )
    
    # calculate the total electric field
    E = create_total_field(V, Eb, Esh)
    save_field(mesh_data.mesh, V, E, problem_params.degree, out_folder, "E.bp")

    results = compute_efficiencies(
        mesh_data,
        tags,
        mesh_params,
        problem_params,
        constants,
        D,
        dx,
        Esh,
        E,
    )
    print_and_check_efficiencies(results)


if __name__ == "__main__":
    main()
