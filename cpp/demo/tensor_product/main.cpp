// Copyright (C) 2022 Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <basix/finite-element.h>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/Constant.h>

using namespace dolfinx;

int main(int argc, char* argv[])
{
  init_logging(argc, argv);
  MPI_Init(&argc, &argv);

  {
    using T = double;
    using U = typename dolfinx::scalar_value_type_t<T>;

    MPI_Comm comm = MPI_COMM_WORLD;

    // Create a Basix continuous Lagrange element of degree 1 with tensor
    // product ordering
    std::vector<int> dof_ordering = {0, 2, 1, 3};
    auto e = std::make_shared<basix::FiniteElement<U>>(basix::create_element<U>(
        basix::element::family::P,
        mesh::cell_type_to_basix_type(mesh::CellType::quadrilateral), 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false, dof_ordering));

    // Create coordinate element
    dolfinx::fem::CoordinateElement<U> coord_element(e);
    auto dofs = e->dof_ordering();

    // Create mesh
    // MPI_Comm comm, std::array<std::array<double, 2>, 2> p,
    //  std::array<std::size_t, 2> n,
    //  const fem::CoordinateElement<T>& coordinate_element,
    //  CellPartitionFunction partitioner
    std::array<std::array<U, 2>, 2> p{0.0, 0.0, 1.0, 1.0};
    std::array<std::size_t, 2> n{2, 2};

    auto part = mesh::create_cell_partitioner(mesh::GhostMode::none);
    auto mesh = std::make_shared<mesh::Mesh<U>>(
        mesh::create_rectangle(comm, p, n, coord_element, part));

    // Create a scalar function space
    auto V = std::make_shared<fem::FunctionSpace<U>>(
        dolfinx::fem::create_functionspace<U>(mesh, *e));

    // get element from function space, and print dof ordering
    auto element = V->element();
    auto new_dof_ordering = element->basix_element().dof_ordering();
    for (std::size_t i = 0; i < dof_ordering.size(); ++i)
      std::cout << dof_ordering[i] << std::endl;

    return 0;

    // Create a finite element Function
    auto u = std::make_shared<fem::Function<T>>(V);

    // interpolate x[0] in the scalar space
    u->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f(x.extent(1));
          for (std::size_t p = 0; p < x.extent(1); ++p)
            f[p] = x(0, p);
          return {f, {f.size()}};
        });

#ifdef HAS_ADIOS2
    // Write the function to a VTX file for visualisation, e.g. using
    // ParaView
    io::VTXWriter<U> outfile(mesh->comm(), "out.bp", {u}, "BP4");
    outfile.write(0.0);
    outfile.close();
#endif

  // print x array
  auto array = u->x()->array();
  for (std::size_t i = 0; i < array.size(); ++i)
    std::cout << array[i] << std::endl;

  }

  MPI_Finalize();

  return 0;
}
