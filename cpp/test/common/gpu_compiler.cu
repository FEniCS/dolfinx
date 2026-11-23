// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

// Compiled as CUDA (nvcc) or HIP, as GPU code built against DOLFINx
// is: the DOLFINx headers must pass the GPU compiler's host and device
// front ends. dolfinx_headers.h (generated) includes every installed
// DOLFINx header. Called from the host-compiled test in
// gpu_compiler.cpp.

#include "dolfinx_headers.h"
#include "gpu_compiler.h"
#ifdef __HIP__
#include <hip/hip_runtime.h>
#endif
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
/// Create a degree `degree` Lagrange space on `mesh`.
std::shared_ptr<fem::FunctionSpace<double>>
create_p_space(std::shared_ptr<mesh::Mesh<double>> mesh, basix::cell::type cell,
               int degree)
{
  basix::FiniteElement element = basix::create_element<double>(
      basix::element::family::P, cell, degree,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);
  return std::make_shared<fem::FunctionSpace<double>>(
      fem::create_functionspace<double>(
          mesh, std::make_shared<fem::FiniteElement<double>>(
                    element, mesh->geometry().dim())));
}

/// Interpolate x0 + x1 into P1 and check it at the dof coordinates.
bool check_interpolation(std::shared_ptr<mesh::Mesh<double>> mesh)
{
  std::shared_ptr<fem::FunctionSpace<double>> V
      = create_p_space(mesh, basix::cell::type::triangle, 1);
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
bool check_submesh(const mesh::Mesh<double>& mesh)
{
  const int tdim = mesh.topology()->dim();
  std::vector<std::int32_t> cells
      = mesh::locate_entities(mesh, tdim,
                              [](auto x)
                              {
                                std::vector<std::int8_t> marker(x.extent(1));
                                for (std::size_t p = 0; p < x.extent(1); ++p)
                                  marker[p] = x(0, p) <= 0.5;
                                return marker;
                              });
  auto [submesh, cell_map, vertex_map, x_map]
      = mesh::create_submesh(mesh, tdim, cells);
  std::vector<std::int32_t> subcells(
      submesh.topology()->index_map(tdim)->size_local());
  std::iota(subcells.begin(), subcells.end(), 0);
  std::vector<std::int32_t> parent_cells = cell_map.sub_topology_to_topology(
      std::span<const std::int32_t>(subcells), false);
  return cell_map.sub_topology_to_topology(
             std::span<const std::int32_t>(parent_cells), true)
         == subcells;
}

/// Assemble the FFCx Poisson stiffness matrix (P2, tetrahedra).
/// @return Whether a non-zero matrix annihilates constants, and whether
/// constrained rows have unit diagonal and zero off-diagonal entries.
std::pair<bool, bool> check_assembly(MPI_Comm comm)
{
  auto mesh = std::make_shared<mesh::Mesh<double>>(
      mesh::create_box(comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {3, 3, 3},
                       mesh::CellType::tetrahedron));
  std::shared_ptr<fem::FunctionSpace<double>> V
      = create_p_space(mesh, basix::cell::type::tetrahedron, 2);
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
  std::span<const double> y_owned
      = std::span<const double>(y.array()).first(A.index_map(0)->size_local());
  const bool nonzero = A.squared_norm() > 0.0;
  bool annihilates_constants = std::ranges::all_of(
      y_owned, [](double v) { return std::abs(v) < 1e-12; });

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

  const auto [dofs, num_owned] = bc.dof_indices();
  bool constrained_rows = true;
  for (std::int32_t row : dofs.first(num_owned))
  {
    bool has_diagonal = false;
    for (std::int64_t j = A_bc.row_ptr()[row]; j < A_bc.row_ptr()[row + 1]; ++j)
    {
      const bool diagonal = A_bc.cols()[j] == row;
      has_diagonal = has_diagonal or diagonal;
      constrained_rows
          = constrained_rows
            and std::abs(A_bc.values()[j] - (diagonal ? 1.0 : 0.0)) < 1e-12;
    }
    constrained_rows = constrained_rows and has_diagonal;
  }

  return {nonzero and annihilates_constants, constrained_rows};
}
} // namespace

//-----------------------------------------------------------------------------
dolfinx_test::GpuCompilerResults
dolfinx_test::gpu_compiler_host_code(MPI_Comm comm)
{
  auto mesh = std::make_shared<mesh::Mesh<double>>(mesh::create_rectangle(
      comm, {{{0.0, 0.0}, {1.0, 1.0}}}, {8, 8}, mesh::CellType::triangle));
  auto [assembly, assembly_with_bc] = check_assembly(comm);
  return {.interpolation = check_interpolation(mesh),
          .submesh = check_submesh(*mesh),
          .assembly = assembly,
          .assembly_with_bc = assembly_with_bc};
}
//-----------------------------------------------------------------------------
