
// Copyright (C) 2022 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <basix/e-lagrange.h>
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
#include <xtensor/xmath.hpp>

using namespace dolfinx;

/// This program shows how to create finite element spaces without FFCx
/// generated code
int main(int argc, char* argv[])
{
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);

  {
    // Create mesh
    auto mesh = std::make_shared<mesh::Mesh>(mesh::create_rectangle(
        MPI_COMM_WORLD, {{{0.0, 0.0}, {2.0, 1.0}}}, {32, 16},
        mesh::CellType::triangle, mesh::GhostMode::none));

    // Create a Basix element
    basix::FiniteElement e_basix = basix::element::create_lagrange(
        mesh::cell_type_to_basix_type(mesh::CellType::triangle), 1,
        basix::element::lagrange_variant::equispaced, false);

    // Create a DOLFINx element
    auto e = std::make_shared<fem::FiniteElement>(e_basix, 1);

    // Create a dofmap
    auto layout = std::make_shared<fem::ElementDofLayout>(
        1, e_basix.entity_dofs(), e_basix.entity_closure_dofs(),
        std::vector<int>(),
        std::vector<std::shared_ptr<const fem::ElementDofLayout>>{});

    auto dofmap = std::make_shared<fem::DofMap>(fem::create_dofmap(
        MPI_COMM_WORLD, layout, mesh->topology(), nullptr, e));

    // Create a function space
    auto V = std::make_shared<fem::FunctionSpace>(mesh, e, dofmap);

    // Create a Function
    auto u = std::make_shared<fem::Function<double>>(V);

    // Interpolate and expression the the finite element space and save
    // the result to file
    constexpr double PI = xt::numeric_constants<double>::PI;
    u->interpolate([PI](auto& x) { return xt::sin(2 * PI * xt::row(x, 0)); });

    // Save solution in VTK format
    io::VTKFile file(MPI_COMM_WORLD, "u.pvd", "w");
    file.write({*u}, 0.0);

#ifdef HAS_ADIOS2
    io::VTXWriter outfile(MPI_COMM_WORLD, "output.bp", {u});
    outfile.write(0.0);
    outfile.close();
#endif
  }

  common::subsystem::finalize_mpi();
  return 0;
}
