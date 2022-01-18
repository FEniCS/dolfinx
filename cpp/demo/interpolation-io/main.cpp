
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

#include <xtensor/xio.hpp>

using namespace dolfinx;
using T = PetscScalar;

void interpolate_scalar(const std::shared_ptr<mesh::Mesh>& mesh)
{
  // Create a Basix continuous Lagrange element of degree 1
  basix::FiniteElement e_basix = basix::element::create_lagrange(
      mesh::cell_type_to_basix_type(mesh::CellType::triangle), 1,
      basix::element::lagrange_variant::equispaced, true);

  // Create a DOLFINx scalar Lagrange element
  auto e = std::make_shared<fem::FiniteElement>(e_basix, 1);

  // Create a dofmap
  fem::ElementDofLayout layout(1, e_basix.entity_dofs(),
                               e_basix.entity_closure_dofs(), {}, {});
  auto dofmap = std::make_shared<fem::DofMap>(
      fem::create_dofmap(mesh->comm(), layout, mesh->topology(), nullptr, e));

  // Create a function space
  auto V = std::make_shared<fem::FunctionSpace>(mesh, e, dofmap);

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

void interpolate_nedelec(const std::shared_ptr<mesh::Mesh>& mesh)
{
  // Create a Basix Nedelec (first kind) element of degree 1 (lowest
  // order, dim=3 on triangle)
  basix::FiniteElement e_basix = basix::element::create_nedelec(
      mesh::cell_type_to_basix_type(mesh::CellType::triangle), 1, false);

  // Create a DOLFINx Nedelec element
  auto e = std::make_shared<fem::FiniteElement>(e_basix, 1);

  // Create a dofmap
  fem::ElementDofLayout layout(1, e_basix.entity_dofs(),
                               e_basix.entity_closure_dofs(), {}, {});
  auto dofmap = std::make_shared<fem::DofMap>(
      fem::create_dofmap(mesh->comm(), layout, mesh->topology(), nullptr, e));

  // Create a function space
  auto V = std::make_shared<fem::FunctionSpace>(mesh, e, dofmap);

  // Create a finite element Function
  auto u = std::make_shared<fem::Function<T>>(V);

  // Interpolate the vector field
  //  u = [0, 1]  if x[0] < 0.5
  //      [1, 1]  if x[0] >= 0.5
  // in the Nedelec space.
  //
  // Note that the x1 component of this field is continuous, and the x0
  // component is discontinuous across x0 = 0.5. This function lies in
  // the Nedelec space when there are cell edges aligned to x0 = 0.5.
  u->interpolate(
      [](auto& x) -> xt::xtensor<T, 2>
      {
        xt::xtensor<T, 2> v = xt::ones<T>({std::size_t(2), x.shape(1)});
        auto v0 = xt::row(v, 0);
        auto x0 = xt::row(x, 0);
        xt::row(v, 0) = xt::where(x0 < 0.5, 0, 1);
        return v;
      });

  // Nedelec spaces are not generally supported by visualisation tools.
  // Simply evaluting a Nedelec function at cell vertices can
  // mis-represent the function. However, we can represented a Nedelec
  // function exactly in a discontinuous Lagrange space which we can
  // then visualise. We do this here.

  // First create a degree 1 vector-valued discontinuous Lagrange space:
  basix::FiniteElement e_basix_l = basix::element::create_lagrange(
      mesh::cell_type_to_basix_type(mesh::CellType::triangle), 1,
      basix::element::lagrange_variant::equispaced, true);
  auto e_l = std::make_shared<fem::FiniteElement>(e_basix_l, 2);
  fem::ElementDofLayout layout_l(2, e_basix_l.entity_dofs(),
                                 e_basix_l.entity_closure_dofs(), {}, {});
  auto dofmap_l = std::make_shared<fem::DofMap>(fem::create_dofmap(
      mesh->comm(), layout_l, mesh->topology(), nullptr, e_l));
  auto V_l = std::make_shared<fem::FunctionSpace>(mesh, e_l, dofmap_l);
  auto u_l = std::make_shared<fem::Function<T>>(V_l);

  // Interpolate the Nedelec function into the discontinuous Lagrange
  // space:
  u_l->interpolate(*u);

  // Outout the discontinuous Lagrange space in VTK format. When
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

  {
    // Create mesh
    auto mesh = std::make_shared<mesh::Mesh>(mesh::create_rectangle(
        MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}}, {32, 32},
        mesh::CellType::triangle, mesh::GhostMode::none));

    interpolate_scalar(mesh);

    interpolate_nedelec(mesh);
  }

  common::subsystem::finalize_mpi();
  return 0;
}
