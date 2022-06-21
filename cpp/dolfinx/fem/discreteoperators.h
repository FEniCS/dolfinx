// Copyright (C) 2015-2022 Garth N. Wells, Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "DofMap.h"
#include "FiniteElement.h"
#include "FunctionSpace.h"
#include <array>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/math.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xtensor.hpp>
#include <xtl/xspan.hpp>

namespace dolfinx::fem
{

/// @brief Assemble a discrete gradient operator.
///
/// The discrete gradient operator \f$A\f$ interpolates the gradient of
/// a Lagrange finite element function in \f$V_0 \subset H^1\f$ into a
/// Nédélec (first kind) space \f$V_1 \subset H({\rm curl})\f$, i.e.
/// \f$\nabla V_0 \rightarrow V_1\f$. If \f$u_0\f$ is the
/// degree-of-freedom vector associated with \f$V_0\f$, the hen
/// \f$u_1=Au_0\f$ where \f$u_1\f$ is the degrees-of-freedom vector for
/// interpolating function in the \f$H({\rm curl})\f$ space. An example
/// of where discrete gradient operators are used is the creation of
/// algebraic multigrid solvers for \f$H({\rm curl})\f$  and
/// \f$H({\rm div})\f$ problems.
///
/// @note The sparsity pattern for a discrete operator can be
/// initialised using sparsitybuild::cells. The space `V1` should be
/// used for the rows of the sparsity pattern, `V0` for the columns.
///
/// @warning This function relies on the user supplying appropriate
/// input and output spaces. See parameter descriptions.
///
/// @param[in] V0 A Lagrange space to interpolate the gradient from
/// @param[in] V1 A Nédélec (first kind) space to interpolate into
/// @param[in] mat_set A functor that sets values in a matrix
template <typename T, typename U>
void discrete_gradient(const fem::FunctionSpace& V0,
                       const fem::FunctionSpace& V1, U&& mat_set)
{
  // Get mesh
  std::shared_ptr<const mesh::Mesh> mesh = V1.mesh();
  assert(mesh);

  // Check spaces
  std::shared_ptr<const FiniteElement> e0 = V0.element();
  assert(e0);
  if (e0->map_type() != basix::maps::type::identity)
    throw std::runtime_error("Wrong finite element space for V0.");
  if (e0->block_size() != 1)
    throw std::runtime_error("Block size is greather than 1 for V0.");
  if (e0->reference_value_size() != 1)
    throw std::runtime_error("Wrong value size for V0.");

  std::shared_ptr<const FiniteElement> e1 = V1.element();
  assert(e1);
  if (e1->map_type() != basix::maps::type::covariantPiola)
    throw std::runtime_error("Wrong finite element space for V1.");
  if (e1->block_size() != 1)
    throw std::runtime_error("Block size is greather than 1 for V1.");

  // Get V0 (H(curl)) space interpolation points
  const xt::xtensor<double, 2> X = e1->interpolation_points();

  // Tabulate first order derivatives of Lagrange space at H(curl)
  // interpolation points
  const int ndofs0 = e0->space_dimension();
  const int tdim = mesh->topology().dim();
  xt::xtensor<double, 4> phi0
      = xt::empty<double>({tdim + 1, int(X.shape(0)), ndofs0, 1});
  e0->tabulate(phi0, X, 1);

  // Reshape lagrange basis derivatives as a matrix of shape (tdim *
  // num_points, num_dofs_per_cell)
  auto dphi0 = xt::view(phi0, xt::xrange(std::size_t(1), phi0.shape(0)),
                        xt::all(), xt::all(), 0);
  auto dphi_reshaped
      = xt::reshape_view(dphi0, {tdim * phi0.shape(1), phi0.shape(2)});

  // Get inverse DOF transform function
  auto apply_inverse_dof_transform
      = e1->get_dof_transformation_function<T>(true, true, false);

  // Generate cell permutations
  mesh->topology_mutable().create_entity_permutations();
  const std::vector<std::uint32_t>& cell_info
      = mesh->topology().get_cell_permutation_info();

  // Create element kernel function
  std::shared_ptr<const DofMap> dofmap0 = V0.dofmap();
  assert(dofmap0);
  std::shared_ptr<const DofMap> dofmap1 = V1.dofmap();
  assert(dofmap1);

  // Build the element interpolation matrix
  std::vector<T> A(e1->space_dimension() * ndofs0);
  {
    auto _A = xt::adapt(A, std::vector<int>{e1->space_dimension(), ndofs0});
    const xt::xtensor<double, 2> Pi = e1->interpolation_operator();
    math::dot(Pi, dphi_reshaped, _A);
  }

  // Insert local interpolation matrix for each cell
  auto cell_map = mesh->topology().index_map(tdim);
  assert(cell_map);
  std::int32_t num_cells = cell_map->size_local();
  std::vector<T> Ae(A.size());
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    std::copy(A.cbegin(), A.cend(), Ae.begin());
    apply_inverse_dof_transform(Ae, cell_info, c, ndofs0);
    mat_set(dofmap1->cell_dofs(c), dofmap0->cell_dofs(c), Ae);
  }
}

/// @brief Assemble an interpolation operator matrix
///
/// The interpolation operator \f$A\f$ interpolates a function in the
/// space \f$V_0\f$ into a space \f$V_1\f$. If \f$u_0\f$ is the
/// degree-of-freedom vector associated with \f$V_0\f$, then the
/// degree-of-freedom vector \f$u_1\f$ for the interpolated function in
/// \f$V_1\f$ is given by \f$u_1=Au_0\f$.
///
/// @note The sparsity pattern for a discrete operator can be
/// initialised using sparsitybuild::cells. The space `V1` should be
/// used for the rows of the sparsity pattern, `V0` for the columns.
///
/// @param[in] V0 The space to interpolate from
/// @param[in] V1 The space to interpolate to
/// @param[in] mat_set A functor that sets values in a matrix
template <typename T, typename U>
void interpolation_matrix(const fem::FunctionSpace& V0,
                          const fem::FunctionSpace& V1, U&& mat_set)
{
  // Get mesh
  auto mesh = V0.mesh();
  assert(mesh);

  // Mesh dims
  const int tdim = mesh->topology().dim();
  const int gdim = mesh->geometry().dim();

  // Get elements
  std::shared_ptr<const FiniteElement> element0 = V0.element();
  assert(element0);
  std::shared_ptr<const FiniteElement> element1 = V1.element();
  assert(element1);

  xtl::span<const std::uint32_t> cell_info;
  if (element1->needs_dof_transformations()
      or element0->needs_dof_transformations())
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }

  // Get dofmaps
  auto dofmap0 = V0.dofmap();
  auto dofmap1 = V1.dofmap();

  // Get block sizes and dof transformation operators
  const int bs0 = element0->block_size();
  const int bs1 = element1->block_size();
  const auto apply_dof_transformation0
      = element0->get_dof_transformation_function<double>(false, false, false);
  const auto apply_inverse_dof_transform1
      = element1->get_dof_transformation_function<T>(true, true, false);

  // Get sizes of elements
  const std::size_t space_dim0 = element0->space_dimension();
  const std::size_t space_dim1 = element1->space_dimension();
  const std::size_t dim0 = space_dim0 / bs0;
  const std::size_t value_size_ref0 = element0->reference_value_size() / bs0;
  const std::size_t value_size0 = element0->value_size() / bs0;
  const std::size_t value_size1 = element1->value_size() / bs1;

  // Get geometry data
  const fem::CoordinateElement& cmap = mesh->geometry().cmap();
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  const std::size_t num_dofs_g = cmap.dim();
  xtl::span<const double> x_g = mesh->geometry().x();

  // Evaluate coordinate map basis at reference interpolation points
  const xt::xtensor<double, 2> X = element1->interpolation_points();
  xt::xtensor<double, 4> phi(cmap.tabulate_shape(1, X.shape(0)));
  cmap.tabulate(1, X, phi);
  xt::xtensor<double, 2> dphi
      = xt::view(phi, xt::range(1, tdim + 1), 0, xt::all(), 0);

  // Evaluate V0 basis functions at reference interpolation points for V1
  xt::xtensor<double, 4> basis_derivatives_reference0(
      {1, X.shape(0), dim0, value_size_ref0});
  element0->tabulate(basis_derivatives_reference0, X, 0);

  double rtol = 1e-14;
  double atol = 1e-14;
  auto inds = xt::isclose(basis_derivatives_reference0, 0.0, rtol, atol);
  xt::filtration(basis_derivatives_reference0, inds) = 0.0;

  // Create working arrays
  xt::xtensor<double, 3> basis_reference0({X.shape(0), dim0, value_size_ref0});
  xt::xtensor<double, 3> J({X.shape(0), gdim, tdim});
  xt::xtensor<double, 3> K({X.shape(0), tdim, gdim});
  std::vector<double> detJ(X.shape(0));

  // Get the interpolation operator (matrix) `Pi` that maps a function
  // evaluated at the interpolation points to the element degrees of
  // freedom, i.e. dofs = Pi f_x
  const xt::xtensor<double, 2>& Pi_1 = element1->interpolation_operator();
  bool interpolation_ident = element1->interpolation_ident();

  using u_t = xt::xview<decltype(basis_reference0)&, std::size_t,
                        xt::xall<std::size_t>, xt::xall<std::size_t>>;
  using U_t = xt::xview<decltype(basis_reference0)&, std::size_t,
                        xt::xall<std::size_t>, xt::xall<std::size_t>>;
  using J_t = xt::xview<decltype(J)&, std::size_t, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  using K_t = xt::xview<decltype(K)&, std::size_t, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  auto push_forward_fn0 = element0->map_fn<u_t, U_t, J_t, K_t>();

  // Basis values of Lagrange space unrolled for block size
  // (num_quadrature_points, Lagrange dof, value_size)
  xt::xtensor<double, 3> basis_values = xt::zeros<double>(
      {X.shape(0), bs0 * dim0, (std::size_t)element1->value_size()});
  xt::xtensor<double, 3> mapped_values(
      {X.shape(0), bs0 * dim0, (std::size_t)element1->value_size()});

  using u1_t = xt::xview<decltype(basis_values)&, std::size_t,
                         xt::xall<std::size_t>, xt::xall<std::size_t>>;
  using U1_t = xt::xview<decltype(mapped_values)&, std::size_t,
                         xt::xall<std::size_t>, xt::xall<std::size_t>>;
  auto pull_back_fn1 = element1->map_fn<U1_t, u1_t, K_t, J_t>();

  xt::xtensor<double, 2> coordinate_dofs({num_dofs_g, 3});
  xt::xtensor<double, 3> basis0({X.shape(0), dim0, value_size0});
  std::vector<T> A(space_dim0 * space_dim1);
  std::vector<T> local1(space_dim1);

  std::vector<std::size_t> shape
      = {X.shape(0), (std::size_t)element1->value_size(), space_dim0};
  auto _A = xt::adapt(A, shape);

  // Iterate over mesh and interpolate on each cell
  auto cell_map = mesh->topology().index_map(tdim);
  assert(cell_map);
  std::int32_t num_cells = cell_map->size_local();
  auto _coordinate_dofs
      = xt::view(coordinate_dofs, xt::all(), xt::xrange(0, gdim));
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    // Get cell geometry (coordinate dofs)
    auto x_dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      common::impl::copy_N<3>(std::next(x_g.begin(), 3 * x_dofs[i]),
                              std::next(coordinate_dofs.begin(), 3 * i));
    }

    // Compute Jacobians and reference points for current cell
    J.fill(0);
    for (std::size_t p = 0; p < X.shape(0); ++p)
    {
      auto _J = xt::view(J, p, xt::all(), xt::all());
      cmap.compute_jacobian(dphi, _coordinate_dofs, _J);
      cmap.compute_jacobian_inverse(_J, xt::view(K, p, xt::all(), xt::all()));
      detJ[p] = cmap.compute_jacobian_determinant(_J);
    }

    // Get evaluated basis on reference, apply DOF transformations, and
    // push forward to physical element
    basis_reference0 = xt::view(basis_derivatives_reference0, 0, xt::all(),
                                xt::all(), xt::all());
    for (std::size_t p = 0; p < X.shape(0); ++p)
    {
      apply_dof_transformation0(
          xtl::span(basis_reference0.data() + p * dim0 * value_size_ref0,
                    dim0 * value_size_ref0),
          cell_info, c, value_size_ref0);
    }

    for (std::size_t i = 0; i < basis0.shape(0); ++i)
    {
      auto _K = xt::view(K, i, xt::all(), xt::all());
      auto _J = xt::view(J, i, xt::all(), xt::all());
      auto _u = xt::view(basis0, i, xt::all(), xt::all());
      auto _U = xt::view(basis_reference0, i, xt::all(), xt::all());
      push_forward_fn0(_u, _U, _J, detJ[i], _K);
    }

    // Unroll basis function for input space for block size
    for (std::size_t p = 0; p < X.shape(0); ++p)
      for (std::size_t i = 0; i < dim0; ++i)
        for (std::size_t j = 0; j < value_size0; ++j)
          for (int k = 0; k < bs0; ++k)
            basis_values(p, i * bs0 + k, j * bs0 + k) = basis0(p, i, j);

    // Pull back the physical values to the reference of output space
    for (std::size_t p = 0; p < basis_values.shape(0); ++p)
    {
      auto _K = xt::view(K, p, xt::all(), xt::all());
      auto _J = xt::view(J, p, xt::all(), xt::all());
      auto _u = xt::view(basis_values, p, xt::all(), xt::all());
      auto _U = xt::view(mapped_values, p, xt::all(), xt::all());
      pull_back_fn1(_U, _u, _K, 1.0 / detJ[p], _J);
    }

    // Apply interpolation matrix to basis values of V0 at the interpolation
    // points of V1
    if (interpolation_ident)
      _A.assign(xt::transpose(mapped_values, {0, 2, 1}));
    else
    {
      for (std::size_t i = 0; i < mapped_values.shape(1); ++i)
      {
        auto _mapped_values = xt::view(mapped_values, xt::all(), i, xt::all());
        impl::interpolation_apply(Pi_1, _mapped_values, local1, bs1);
        for (std::size_t j = 0; j < local1.size(); j++)
          A[space_dim0 * j + i] = local1[j];
      }
    }
    apply_inverse_dof_transform1(A, cell_info, c, space_dim0);
    mat_set(dofmap1->cell_dofs(c), dofmap0->cell_dofs(c), A);
  }
}

} // namespace dolfinx::fem
