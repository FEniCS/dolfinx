// ```text
// Copyright (C) 2022 Igor A. Baratta and Massimiliano Leoni
// This file is part of DOLFINx (https://www.fenicsproject.org)
// SPDX-License-Identifier:    LGPL-3.0-or-later
// ```

// # Interpolation different meshes

#include <basix/e-lagrange.h>
#include <dolfinx/fem/dolfinx_fem.h>
#include <dolfinx/io/ADIOS2Writers.h>
#include <dolfinx/mesh/generation.h>
#include <memory>

using namespace dolfinx;
using T = double;

int main(int argc, char* argv[])
{
  init_logging(argc, argv);
  MPI_Init(&argc, &argv);
  {
    MPI_Comm comm = MPI_COMM_WORLD;

    // Create a tetrahedral mesh
    auto mesh_tet = std::make_shared<mesh::Mesh<double>>(
        mesh::create_box(comm, {{{0, 0, 0}, {1, 1, 1}}}, {20, 20, 20},
                         mesh::CellType::tetrahedron));

    // Create a hexahedral mesh
    auto mesh_hex = std::make_shared<mesh::Mesh<double>>(
        mesh::create_box(comm, {{{0, 0, 0}, {1, 1, 1}}}, {15, 15, 15},
                         mesh::CellType::hexahedron));

    basix::FiniteElement element_tet = basix::element::create_lagrange<double>(
        mesh::cell_type_to_basix_type(mesh_tet->topology()->cell_type()), 1,
        basix::element::lagrange_variant::equispaced, false);
    auto V_tet = std::make_shared<fem::FunctionSpace<double>>(
        fem::create_functionspace<double>(
            mesh_tet,
            std::make_shared<fem::FiniteElement<double>>(element_tet, 3)));

    basix::FiniteElement element_hex = basix::element::create_lagrange<double>(
        mesh::cell_type_to_basix_type(mesh_hex->topology()->cell_type()), 2,
        basix::element::lagrange_variant::equispaced, false);
    auto V_hex = std::make_shared<fem::FunctionSpace<double>>(
        fem::create_functionspace<double>(
            mesh_hex,
            std::make_shared<fem::FiniteElement<double>>(element_hex, 3)));

    auto u_tet = std::make_shared<fem::Function<T>>(V_tet);
    auto u_hex = std::make_shared<fem::Function<T>>(V_hex);

    auto fun = [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
    {
      std::vector<T> fdata(3 * x.extent(1), 0.0);
      using dextent = MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>;
      MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<double, dextent> f(fdata.data(), 3,
                                                                x.extent(1));
      for (std::size_t i = 0; i < x.extent(1); ++i)
      {
        f(0, i) = std::cos(10 * x(0, i)) * std::sin(10 * x(2, i));
        f(1, i) = std::sin(10 * x(0, i)) * std::sin(10 * x(2, i));
        f(2, i) = std::cos(10 * x(0, i)) * std::cos(10 * x(2, i));
      }
      return {std::move(fdata), {3, x.extent(1)}};
    };

    // Interpolate an expression into u_tet
    u_tet->interpolate(fun);

    // Interpolate from u_tet to u_hex
    auto cell_map
        = mesh_hex->topology()->index_map(mesh_hex->topology()->dim());
    assert(cell_map);
    std::vector<std::int32_t> cells(
        cell_map->size_local() + cell_map->num_ghosts(), 0);
    std::iota(cells.begin(), cells.end(), 0);
    geometry::PointOwnershipData<T> interpolation_data
        = fem::create_interpolation_data(
            u_hex->function_space()->mesh()->geometry(),
            *u_hex->function_space()->element(),
            *u_tet->function_space()->mesh(), std::span(cells), 1e-8);
    u_hex->interpolate(*u_tet, cells, interpolation_data);

#ifdef HAS_ADIOS2
    io::VTXWriter<double> write_tet(mesh_tet->comm(), "u_tet.bp", {u_tet});
    write_tet.write(0.0);
    io::VTXWriter<double> write_hex(mesh_hex->comm(), "u_hex.bp", {u_hex});
    write_hex.write(0.0);
#endif
  }
  MPI_Finalize();

  return 0;
}
