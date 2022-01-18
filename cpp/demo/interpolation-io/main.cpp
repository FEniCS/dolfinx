
// Copyright (C) 2022 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <basix/e-lagrange.h>
#include <basix/e-nedelec.h>
#include <cmath>
#include <dolfinx/common/subsystem.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/io/ADIOS2Writers.h>
#include <dolfinx/io/VTKFile.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/generation.h>
#include <petscsys.h>
#include <xtensor/xmath.hpp>

using namespace dolfinx;
using T = PetscScalar;

// This function interpolations a function is a finite element space and
// outputs the finite element function to a VTK file for visualisation.
// It also shows how to create a finite element using Basix.
void interpolate_scalar(const std::shared_ptr<mesh::Mesh>& mesh)
{
  // Create a Basix continuous Lagrange element of degree 1
  basix::FiniteElement e = basix::element::create_lagrange(
      mesh::cell_type_to_basix_type(mesh::CellType::triangle), 1,
      basix::element::lagrange_variant::equispaced, true);

  // Create a scalar function space
  auto V = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(mesh, e, 1));

  // Create a finite element Function
  auto u = std::make_shared<fem::Function<T>>(V);

  // Interpolate sin(2 \pi x[0]) in the scalar Lagrange finite element
  // space
  constexpr double PI = xt::numeric_constants<double>::PI;
  u->interpolate([PI](auto& x) { return xt::sin(2 * PI * xt::row(x, 0)); });

  // Write the function to a VTK file for visualisation, e.g. using
  // ParaView
  io::VTKFile file(MPI_COMM_WORLD, "u.pvd", "w");
  file.write({*u}, 0.0);
}

// This function interpolations a function is a H(curl) finite element
// space. To visualise the function, it interpolates the H(curl) finite
// element function in a discontinuous Lagrange space and outputs the
// Lagrange finite element function to a VTX file for visualisation.
void interpolate_nedelec(const std::shared_ptr<mesh::Mesh>& mesh)
{
  // Create a Basix Nedelec (first kind) element of degree 2 (lowest
  // order, dim=6 on triangle)
  basix::FiniteElement e = basix::element::create_nedelec(
      mesh::cell_type_to_basix_type(mesh::CellType::triangle), 2, false);

  // Create a function space
  auto V = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(mesh, e, 1));

  // Create a finite element Function
  auto u = std::make_shared<fem::Function<T>>(V);

  // Interpolate the vector field
  //  u = [x[0], x[1]]  if x[0] < 0.5
  //      [x[0], x[1] + 1]  if x[0] >= 0.5
  // in the Nedelec space.
  //
  // Note that the x1 component of this field is continuous, and the x0
  // component is discontinuous across x0 = 0.5. This function lies in
  // the Nedelec space when there are cell edges aligned to x0 = 0.5.
  u->interpolate(
      [](auto& x) -> xt::xtensor<T, 2>
      {
        xt::xtensor<T, 2> v = xt::view(x, xt::range(0, 2), xt::all());
        auto v0 = xt::row(v, 0);
        auto x0 = xt::row(x, 0);
        v0 = xt::where(x0 < 0.5, v0, v0 + 1);
        return v;
      });

  // Nedelec spaces are not generally supported by visualisation tools.
  // Simply evaluting a Nedelec function at cell vertices can
  // mis-represent the function. However, we can represented a Nedelec
  // function exactly in a discontinuous Lagrange space which we can
  // then visualise. We do this here.

  // First create a degree 2 vector-valued discontinuous Lagrange space:
  basix::FiniteElement e_l = basix::element::create_lagrange(
      mesh::cell_type_to_basix_type(mesh::CellType::triangle), 2,
      basix::element::lagrange_variant::equispaced, true);

  // Create a function space
  auto V_l = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(mesh, e_l, 2));

  auto u_l = std::make_shared<fem::Function<T>>(V_l);

  // Interpolate the Nedelec function into the discontinuous Lagrange
  // space:
  u_l->interpolate(*u);

  // Output the discontinuous Lagrange space in VTK format. When
  // plotting the x0 component the field will appear discontinuous at x0
  // = 0.5 (jump in the normal component between cells) and the x1
  // component will appear continuous (continuous tangent component
  // between cells).
#ifdef HAS_ADIOS2
  io::VTXWriter outfile(MPI_COMM_WORLD, "output_nedelec.bp", {u_l});
  outfile.write(0.0);
  outfile.close();
#endif
}

/// This program shows how to create finite element spaces without FFCx
/// generated code
int main(int argc, char* argv[])
{
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);

  // The main body of the function is scoped with the curly braces to
  // ensure that all objects that depend on an MPI communicator are
  // destroyed before MPI is finalised at the end of this function.
  {
    // Create mesh
    auto mesh = std::make_shared<mesh::Mesh>(mesh::create_rectangle(
        MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}}, {32, 32},
        mesh::CellType::triangle, mesh::GhostMode::none));

    // Interpolate a function in a scalar Lagrange space and output the
    // result to file for visualisation
    interpolate_scalar(mesh);

    // Interpolate a function in a H(curl) finite element space, and
    // then interpolate the H(curl) in a discontinuous Lagrange space
    // for visualisation
    interpolate_nedelec(mesh);
  }

  common::subsystem::finalize_mpi();
  return 0;
}
