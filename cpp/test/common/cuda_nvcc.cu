// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

// Compiled by nvcc, as CUDA code built against DOLFINx is: the DOLFINx
// headers must pass nvcc's host and device front ends. Called from the
// host-compiled test in cuda.cpp. cuda_headers.cu (generated) includes
// every installed header.

#include "cuda_nvcc.h"
#include "poisson.h"
#include <algorithm>
#include <basix/finite-element.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <dolfinx.h>
#include <dolfinx/la/MatrixCSR.h>
#include <functional>
#include <memory>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

using namespace dolfinx;

// Explicit instantiation compiles every non-template member, not only
// those used below.
#define DOLFINX_TEST_INSTANTIATE(T)                                            \
  template class mesh::Mesh<T>;                                                \
  template class mesh::Geometry<T>;                                            \
  template class fem::CoordinateElement<T>;                                    \
  template class fem::FiniteElement<T>;                                        \
  template class fem::FunctionSpace<T>;                                        \
  template class fem::Function<T>;                                             \
  template class fem::Constant<T>;                                             \
  template class fem::Form<T>;                                                 \
  template class fem::DirichletBC<T>;                                          \
  template class fem::Expression<T>;                                           \
  template class la::Vector<T>;                                                \
  template class la::MatrixCSR<T>;                                             \
  template class geometry::BoundingBoxTree<T>;
DOLFINX_TEST_INSTANTIATE(float)
DOLFINX_TEST_INSTANTIATE(double)
#undef DOLFINX_TEST_INSTANTIATE
template class mesh::MeshTags<std::int32_t>;

namespace dolfinx_test
{
// Device code, so the headers also pass the device front end. External
// linkage so that it is compiled; not launched, so no GPU is needed.
__global__ void axpy(int n, double a, const double* x, double* y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] += a * x[i];
}
} // namespace dolfinx_test

namespace
{
/// Interpolate x0 + x1 into P1 and check it at the dof coordinates.
bool check_interpolation(MPI_Comm comm)
{
  auto mesh = std::make_shared<mesh::Mesh<double>>(mesh::create_rectangle(
      comm, {{{0.0, 0.0}, {1.0, 1.0}}}, {8, 8}, mesh::CellType::triangle));
  basix::FiniteElement element = basix::create_element<double>(
      basix::element::family::P, basix::cell::type::triangle, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);
  auto V = std::make_shared<fem::FunctionSpace<double>>(
      fem::create_functionspace<double>(
          mesh, std::make_shared<fem::FiniteElement<double>>(
                    element, mesh->geometry().dim())));

  fem::Function<double> u(V);
  u.interpolate(
      [](auto x) -> std::pair<std::vector<double>, std::vector<std::size_t>>
      {
        std::vector<double> f(x.extent(1));
        for (std::size_t p = 0; p < x.extent(1); ++p)
          f[p] = x(0, p) + x(1, p);
        return {std::move(f), {f.size()}};
      });

  std::vector<double> x = V->tabulate_dof_coordinates(false);
  std::span<const double> u_x = u.x()->array();
  for (std::size_t i = 0; i < u_x.size(); ++i)
  {
    if (std::abs(u_x[i] - (x[3 * i] + x[3 * i + 1])) > 1e-12)
      return false;
  }
  return true;
}

/// Map sub-mesh cells to the parent mesh and back.
bool check_submesh(MPI_Comm comm)
{
  auto mesh = std::make_shared<mesh::Mesh<double>>(mesh::create_rectangle(
      comm, {{{0.0, 0.0}, {1.0, 1.0}}}, {8, 8}, mesh::CellType::triangle));
  const int tdim = mesh->topology()->dim();
  std::vector<std::int32_t> cells
      = mesh::locate_entities(*mesh, tdim,
                              [](auto x)
                              {
                                std::vector<std::int8_t> marker(x.extent(1));
                                for (std::size_t p = 0; p < x.extent(1); ++p)
                                  marker[p] = x(0, p) <= 0.5;
                                return marker;
                              });
  auto [submesh, cell_map, vertex_map, x_map]
      = mesh::create_submesh(*mesh, tdim, cells);
  std::vector<std::int32_t> subcells(
      submesh.topology()->index_map(tdim)->size_local());
  std::iota(subcells.begin(), subcells.end(), 0);
  std::vector<std::int32_t> parent_cells = cell_map.sub_topology_to_topology(
      std::span<const std::int32_t>(subcells), false);
  return cell_map.sub_topology_to_topology(
             std::span<const std::int32_t>(parent_cells), true)
         == subcells;
}
} // namespace

//-----------------------------------------------------------------------------
dolfinx_test::CudaNvccResults dolfinx_test::cuda_nvcc_host_code(MPI_Comm comm)
{
  CudaNvccResults results{};
  results.interpolation = check_interpolation(comm);
  results.submesh = check_submesh(comm);

  // Assemble the FFCx Poisson stiffness matrix (P2, tetrahedra)
  auto mesh = std::make_shared<mesh::Mesh<double>>(
      mesh::create_box(comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {3, 3, 3},
                       mesh::CellType::tetrahedron));
  basix::FiniteElement element = basix::create_element<double>(
      basix::element::family::P, basix::cell::type::tetrahedron, 2,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);
  auto V = std::make_shared<fem::FunctionSpace<double>>(
      fem::create_functionspace<double>(
          mesh, std::make_shared<fem::FiniteElement<double>>(
                    element, mesh->geometry().dim())));
  auto kappa = std::make_shared<fem::Constant<double>>(2.0);
  fem::Form<double> a = fem::create_form<double, double>(
      *form_poisson_a, {V, V}, {}, {{"kappa", kappa}}, {}, {});
  la::SparsityPattern sp = fem::create_sparsity_pattern(a);
  sp.finalize();

  // Stiffness matrix applied to a constant is zero
  la::MatrixCSR<double> A(sp);
  fem::assemble_matrix(A.mat_add_values(), a, {});
  A.scatter_rev();
  la::Vector<double> x(A.index_map(1), 1);
  la::Vector<double> y(A.index_map(1), 1);
  std::ranges::fill(x.array(), 1.0);
  A.mult(x, y);
  auto y_owned
      = std::span<const double>(y.array()).first(A.index_map(0)->size_local());
  results.assembly = std::ranges::all_of(y_owned, [](double v)
                                         { return std::abs(v) < 1e-12; });

  // Assembly with a Dirichlet condition on the boundary
  const int tdim = mesh->topology()->dim();
  mesh->topology_mutable()->create_connectivity(tdim - 1, tdim);
  std::vector<std::int32_t> facets
      = mesh::exterior_facet_indices(*mesh->topology());
  std::vector<std::int32_t> bdofs = fem::locate_dofs_topological(
      *mesh->topology_mutable(), *V->dofmap(), tdim - 1, facets);
  fem::DirichletBC<double> bc(0.0, bdofs, V);
  la::MatrixCSR<double> A_bc(sp);
  fem::assemble_matrix(A_bc.mat_add_values(), a, {std::cref(bc)});
  fem::set_diagonal<double>(A_bc.mat_set_values(), *V, {std::cref(bc)});
  A_bc.scatter_rev();
  results.assembly_with_bc = A_bc.squared_norm() > 0.0;

  return results;
}
//-----------------------------------------------------------------------------
