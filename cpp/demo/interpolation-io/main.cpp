
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

/// This program shows how to create finite element spaces without FFCx
/// generated code
int main(int argc, char* argv[])
{
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);

  {
    // Create mesh
    // auto mesh = std::make_shared<mesh::Mesh>(mesh::create_rectangle(
    //     MPI_COMM_WORLD, {{{0.0, 0.0}, {2.0, 1.0}}}, {32, 16},
    //     mesh::CellType::triangle, mesh::GhostMode::none));
    auto mesh = std::make_shared<mesh::Mesh>(mesh::create_rectangle(
        MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}}, {32, 32},
        mesh::CellType::triangle, mesh::GhostMode::none));

    // Create a Basix element
    basix::FiniteElement e_basix = basix::element::create_lagrange(
        mesh::cell_type_to_basix_type(mesh::CellType::triangle), 1,
        basix::element::lagrange_variant::equispaced, false);

    basix::FiniteElement e_basix_v = basix::element::create_lagrange(
        mesh::cell_type_to_basix_type(mesh::CellType::triangle), 1,
        basix::element::lagrange_variant::equispaced, true);

    basix::FiniteElement e_basix_n = basix::element::create_nedelec(
        mesh::cell_type_to_basix_type(mesh::CellType::triangle), 1, false);

    // Create a DOLFINx element
    auto e = std::make_shared<fem::FiniteElement>(e_basix, 1);
    auto e_v = std::make_shared<fem::FiniteElement>(e_basix_v, 2);
    auto e_n = std::make_shared<fem::FiniteElement>(e_basix_n, 1);

    // Create a dofmap
    fem::ElementDofLayout layout(1, e_basix.entity_dofs(),
                                 e_basix.entity_closure_dofs(), {}, {});
    fem::ElementDofLayout layout_v(2, e_basix_v.entity_dofs(),
                                   e_basix_v.entity_closure_dofs(), {}, {});
    fem::ElementDofLayout layout_n(1, e_basix_n.entity_dofs(),
                                   e_basix_n.entity_closure_dofs(), {}, {});

    auto dofmap = std::make_shared<fem::DofMap>(fem::create_dofmap(
        MPI_COMM_WORLD, layout, mesh->topology(), nullptr, e));
    auto dofmap_v = std::make_shared<fem::DofMap>(fem::create_dofmap(
        MPI_COMM_WORLD, layout_v, mesh->topology(), nullptr, e_v));
    auto dofmap_n = std::make_shared<fem::DofMap>(fem::create_dofmap(
        MPI_COMM_WORLD, layout_n, mesh->topology(), nullptr, e));

    // Create a function space
    auto V = std::make_shared<fem::FunctionSpace>(mesh, e, dofmap);
    auto V_v = std::make_shared<fem::FunctionSpace>(mesh, e_v, dofmap_v);
    auto V_n = std::make_shared<fem::FunctionSpace>(mesh, e_n, dofmap_n);

    // Create a Function
    auto u = std::make_shared<fem::Function<T>>(V);
    auto u_v = std::make_shared<fem::Function<T>>(V_v);
    auto u_n = std::make_shared<fem::Function<T>>(V_n);

    // Interpolate and expression the the finite element space and save
    // the result to file
    constexpr double PI = xt::numeric_constants<double>::PI;
    u->interpolate([PI](auto& x) { return xt::sin(2 * PI * xt::row(x, 0)); });

    u_n->interpolate(
        [](auto& x) -> xt::xtensor<T, 2>
        {
          xt::xtensor<T, 2> v = xt::ones<T>({std::size_t(2), x.shape(1)});
          auto v0 = xt::row(v, 0);
          auto x0 = xt::row(x, 0);
          xt::row(v, 0) = xt::where(x0 < 0.5, 0, 1);
          return v;
        });

    // Save solution in VTK format
    io::VTKFile file(MPI_COMM_WORLD, "u.pvd", "w");
    file.write({*u}, 0.0);

#ifdef HAS_ADIOS2
    io::VTXWriter outfile(MPI_COMM_WORLD, "output.bp", {u});
    outfile.write(0.0);
    outfile.close();

    u_v->interpolate(*u_n);
    // auto x = u_v->x()->array();
    // for (auto xi : x)
    //   std::cout << xi << std::endl;

    io::VTXWriter outfile_n(MPI_COMM_WORLD, "output_n.bp", {u_v});
    outfile_n.write(0.0);
    outfile_n.close();

#endif
  }

  common::subsystem::finalize_mpi();
  return 0;
}
