// Copyright (C) 2015-2025 Garth N. Wells, Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "DofMap.h"
#include "FiniteElement.h"
#include "FunctionSpace.h"
#include <algorithm>
#include <array>
#include <concepts>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/math.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <span>
#include <vector>

namespace dolfinx::fem
{

/// @brief Assemble a discrete curl operator.
///
/// For vector-valued finite functions \f$u \in V_{0} \f$ and \f$v \in V_{1}\f$,
/// consider the interpolation of the curl of \f$u\f$ in the space \f$V_{1}\f$,
/// i.e.
/// \f$\Pi_{V_{1}}: \nabla \times u \rightarrow v\f$, where \f$\Pi_{V_{1}}\f$
/// is the interpolation operator associated with \f$V_{1}\f$. This
/// interpolation of \f$\nabla \times u\f$ into \f$V_{1}\f$ is properly
/// posed and exact for specific choices of function spaces. If \f$V_{0}\f$
/// is a Nédélec (\f$H({\rm curl})\f$) space of degree \f$k > 1\f$ and
/// \f$V_{1}\f$ is a Raviart-Thomas (\f$H({\rm div})\f$) space of degree of
/// at least \f$k - 1\f$, then the interpolation is exact.
///
/// The implementation of this function exploits the result:
///
/// \f[
///   \hat{\nabla} \times \psi_{C}(\boldsymbol{u}) = \psi_{D}(\nabla \times
///   \boldsymbol{u}),
/// \f]
///
/// where \f$\psi_{C}\f$ is the covariant pull-back (to the reference
/// cell) and \f$\psi_{D}\f$ is the contravariant pull-back. See Ern and
/// Guermond (2021), Finite Elements I, Springer Nature,
/// https://doi.org/10.1007/978-3-030-56341-7 [Corollary 9.9 (Commuting
/// properties)]. Hence, the spaces `V0` and `V1` must used covariant
/// and contravariant maps, respectively.
///
/// This function builds a matrix \f$C\f$ (the 'discrete curl'), which
/// when applied to the degrees-of-freedom of \f$u\f$ gives the
/// degrees-of-freedom of \f$v\f$ such that \f$v = \nabla \times u\f$.
/// If the finite element degree-of-freedom vectors associated with
/// \f$u\f$ and \f$v\f$ are \f$a\f$ and \f$b\f$, respectively, then \f$b
/// = C a\f$, which yields \f$v = \Pi_{V} \nabla \times u\f$. It
/// essentially maps that curl of a function in a degree \f$k > 1\f$
/// Nédélec space into a degree \f$k - 1\f$ Raviart-Thomas space.
///
/// The discerete curl is typically used in constructing algebraic
/// multigrid preconditioners for \f$H({\rm div})\f$, e.g. when using
/// the Hypre Auxiliary-space Divergence Solver (ADS).
///
/// @pre `V0` and `V1` must be vector-valued and in three spatial
/// dimensions.
///
/// @tparam T Scalar type of the mesh and elements.
/// @tparam U Scalar type of the matrix being inserted into. This is
/// usually the same as `T`, but may differ for matrix backends that
/// support only a specific type, e.g. PETSc which support only one
/// scalar type for a build of PETSc.
///
/// @param[in] V0 Space that \f$u\f$ is from. It must be a covariant
/// Piola mapped element. It is normally an \f$H({\rm
/// curl})\f$-conforming Nédélec space.
/// @param[in] V1 Space that \f$v\f$ is from. It must be a contravariant
/// Piola mapped element. It is normally an \f$H({\rm
/// div})\f$-conforming Raviart-Thomas space of one degree lower than
/// `V0`.
/// @param[in] mat_set A functor that sets (not add) values in a matrix
/// \f$C\f$.
template <std::floating_point T, dolfinx::scalar U = T>
void discrete_curl(const FunctionSpace<T>& V0, const FunctionSpace<T>& V1,
                   la::MatSet<U> auto&& mat_set)
{
  namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;

  // Get mesh
  auto mesh = V0.mesh();
  assert(mesh);
  assert(V1.mesh());
  if (mesh != V1.mesh())
    throw std::runtime_error("Meshes must be the same.");

  if (mesh->geometry().dim() != 3)
    throw std::runtime_error("Geometric must be equal to 3..");
  if (mesh->geometry().dim() != mesh->topology()->dim())
  {
    throw std::runtime_error(
        "Geometric and topological dimensions must be equal.");
  }
  constexpr int gdim = 3;

  // Get elements
  std::shared_ptr<const FiniteElement<T>> e0 = V0.element();
  assert(e0);
  if (e0->map_type() != basix::maps::type::covariantPiola)
  {
    throw std::runtime_error(
        "Finite element for parent space must be covariant Piola.");
  }

  std::shared_ptr<const FiniteElement<T>> e1 = V1.element();
  assert(e1);
  if (e1->map_type() != basix::maps::type::contravariantPiola)
  {
    throw std::runtime_error(
        "Finite element for target space must be contracovariant Piola.");
  }

  // Get cell orientation information
  std::span<const std::uint32_t> cell_info;
  if (e1->needs_dof_transformations() or e0->needs_dof_transformations())
  {
    mesh->topology_mutable()->create_entity_permutations();
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }

  // Get dofmaps
  auto dofmap0 = V0.dofmap();
  assert(dofmap0);
  auto dofmap1 = V1.dofmap();
  assert(dofmap1);

  // Get dof transformation operators
  auto apply_dof_transformation0
      = e0->template dof_transformation_fn<T>(doftransform::standard, false);
  auto apply_inverse_dof_transform1 = e1->template dof_transformation_fn<U>(
      doftransform::inverse_transpose, false);

  // Get sizes of elements
  const std::size_t space_dim0 = e0->space_dimension();
  const std::size_t space_dim1 = e1->space_dimension();
  if (e0->reference_value_size() != 3)
    throw std::runtime_error("Value size for parent space should be 3.");
  if (e1->reference_value_size() != 3)
    throw std::runtime_error("Value size for target space should be 3.");

  // Get the V1 reference interpolation points
  const auto [X, Xshape] = e1->interpolation_points();

  // Get/compute geometry map and evaluate at interpolation points
  const CoordinateElement<T>& cmap = mesh->geometry().cmap();
  auto x_dofmap = mesh->geometry().dofmap();
  const std::size_t num_dofs_g = cmap.dim();
  std::span<const T> x_g = mesh->geometry().x();
  std::array<std::size_t, 4> Phi_g_shape = cmap.tabulate_shape(1, Xshape[0]);
  std::vector<T> Phi_g_b(std::reduce(Phi_g_shape.begin(), Phi_g_shape.end(), 1,
                                     std::multiplies{}));
  md::mdspan<const T, md::dextents<std::size_t, 4>> Phi_g(Phi_g_b.data(),
                                                          Phi_g_shape);
  cmap.tabulate(1, X, Xshape, Phi_g_b);

  // Geometry data structures
  std::vector<T> coord_dofs_b(num_dofs_g * gdim);
  md::mdspan<T, md::extents<std::size_t, md::dynamic_extent, 3>> coord_dofs(
      coord_dofs_b.data(), num_dofs_g, gdim);
  std::vector<T> J_b(Xshape[0] * gdim * gdim);
  md::mdspan<T, md::extents<std::size_t, md::dynamic_extent, 3, 3>> J(
      J_b.data(), Xshape[0], gdim, gdim);
  std::vector<T> K_b(Xshape[0] * gdim * gdim);
  md::mdspan<T, md::extents<std::size_t, md::dynamic_extent, 3, 3>> K(
      K_b.data(), Xshape[0], gdim, gdim);
  std::vector<T> detJ(Xshape[0]);
  std::vector<T> det_scratch(2 * gdim * gdim);

  // Evaluate V0 basis function derivatives at reference interpolation
  // points for V1
  const auto [Phi0_b, Phi0_shape] = e0->tabulate(X, Xshape, 1);
  md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent,
                                  md::dynamic_extent, md::dynamic_extent, 3>>
      Phi0(Phi0_b.data(),
           Phi0_shape); // (deriv, pt_idx, phi (dof), comp)

  // Create working arrays, (point, phi (dof), deriv, comp)
  // md::dextents<std::size_t, 4>
  md::extents<std::size_t, md::dynamic_extent, md::dynamic_extent, 3, 3>
      dphi_ext(Phi0.extent(1), Phi0.extent(2), Phi0.extent(0) - 1,
               Phi0.extent(3));
  std::vector<T> dPhi0_b(dphi_ext.extent(0) * dphi_ext.extent(1)
                         * dphi_ext.extent(2) * dphi_ext.extent(3));
  md::mdspan<
      T, md::extents<std::size_t, md::dynamic_extent, md::dynamic_extent, 3, 3>>
      dPhi0(dPhi0_b.data(), dphi_ext);
  std::vector<T> dphi0_int_b(dPhi0.size());
  md::mdspan<
      T, md::extents<std::size_t, md::dynamic_extent, md::dynamic_extent, 3, 3>>
      dphi0_int(dphi0_int_b.data(), dPhi0.extents());
  std::vector<T> dphi0_b(dPhi0.size());
  md::mdspan<
      T, md::extents<std::size_t, md::dynamic_extent, md::dynamic_extent, 3, 3>>
      dphi0(dphi0_b.data(), dPhi0.extents());

  // Get the interpolation operator (matrix) Pi that maps a function
  // evaluated at the interpolation points to the V1 element degrees of
  // freedom, i.e. dofs = Pi f_x
  const auto [Pi1_b, pi_shape] = e1->interpolation_operator();
  md::mdspan<const T, md::dextents<std::size_t, 2>> Pi_1(Pi1_b.data(),
                                                         pi_shape);

  // curl data structure
  std::vector<T> curl_b(dPhi0.extent(0) * dPhi0.extent(1) * dPhi0.extent(3));
  md::mdspan<
      T, md::extents<std::size_t, md::dynamic_extent, md::dynamic_extent, 3>>
      curl(curl_b.data(), dPhi0.extent(0), dPhi0.extent(1),
           dPhi0.extent(3)); // (pt_idx, phi (dof), comp)

  std::vector<U> Ab(space_dim0 * space_dim1);
  std::vector<U> local1(space_dim1);

  // Iterate over mesh and interpolate on each cell
  auto cell_map = mesh->topology()->index_map(gdim);
  assert(cell_map);
  for (std::int32_t c = 0; c < cell_map->size_local(); ++c)
  {
    // Get cell geometry (coordinate dofs)
    auto x_dofs = md::submdspan(x_dofmap, c, md::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
      for (std::size_t j = 0; j < gdim; ++j)
        coord_dofs(i, j) = x_g[3 * x_dofs[i] + j];

    // Compute Jacobians at reference points for current cell
    std::ranges::fill(J_b, 0);
    for (std::size_t p = 0; p < Xshape[0]; ++p)
    {
      auto dPhi_g
          = md::submdspan(Phi_g, std::pair(1, gdim + 1), p, md::full_extent, 0);
      auto _J = md::submdspan(J, p, md::full_extent, md::full_extent);
      cmap.compute_jacobian(dPhi_g, coord_dofs, _J);
      auto _K = md::submdspan(K, p, md::full_extent, md::full_extent);
      cmap.compute_jacobian_inverse(_J, _K);
      detJ[p] = cmap.compute_jacobian_determinant(_J, det_scratch);
    }

    // TODO: re-order loops and/or re-pack Phi0 to allow a simple flat
    // copy?

    // Copy (d)Phi0 (on reference) and apply DOF transformation
    // Phi0:  (deriv, pt_idx, phi (dof), comp)
    // dPhi0: (pt_idx, phi (dov), deriv, comp)
    for (std::size_t p = 0; p < Phi0.extent(1); ++p)           // point
      for (std::size_t phi = 0; phi < Phi0.extent(2); ++phi)   // phi_i
        for (std::size_t d = 0; d < Phi0.extent(3); ++d)       // Comp. of phi
          for (std::size_t dx = 0; dx < dPhi0.extent(2); ++dx) // dx
            dPhi0(p, phi, dx, d) = Phi0(dx + 1, p, phi, d);

    for (std::size_t p = 0; p < dPhi0.extent(0); ++p) // point
    {
      // Size: num_phi * num_derivs * num_components
      std::size_t size = dPhi0.extent(1) * dPhi0.extent(2) * dPhi0.extent(3);
      std::size_t offset = p * size; // Offset for point p

      // Shape: (num_phi , (value_size * num_derivs))
      apply_dof_transformation0(std::span(dPhi0.data_handle() + offset, size),
                                cell_info, c,
                                dPhi0.extent(2) * dPhi0.extent(3));
    }

    // Compute curl
    // dPhi0: (pt_idx, phi_idx, deriv, comp)
    // curl: (pt_idx, phi_idx, comp)
    for (std::size_t p = 0; p < curl.extent(0); ++p) // point
    {
      for (std::size_t i = 0; i < curl.extent(1); ++i) // phi_i
      {
        curl(p, i, 0) = dPhi0(p, i, 1, 2) - dPhi0(p, i, 2, 1);
        curl(p, i, 1) = dPhi0(p, i, 2, 0) - dPhi0(p, i, 0, 2);
        curl(p, i, 2) = dPhi0(p, i, 0, 1) - dPhi0(p, i, 1, 0);
      }
    }

    // Apply interpolation matrix to basis derivatives values of V0 at
    // the interpolation points of V1
    for (std::size_t i = 0; i < curl.extent(1); ++i)
    {
      auto values = md::submdspan(curl, md::full_extent, i, md::full_extent);
      impl::interpolation_apply(Pi_1, values, std::span(local1), 1);
      for (std::size_t j = 0; j < local1.size(); ++j)
        Ab[space_dim0 * j + i] = local1[j];
    }

    apply_inverse_dof_transform1(Ab, cell_info, c, space_dim0);
    mat_set(dofmap1->cell_dofs(c), dofmap0->cell_dofs(c), Ab);
  }
}

/// @brief Assemble a discrete gradient operator.
///
/// The discrete gradient operator \f$A\f$ interpolates the gradient of
/// a Lagrange finite element function in \f$V_0 \subset H^1\f$ into a
/// Nédélec (first kind) space \f$V_1 \subset H({\rm curl})\f$, i.e.
/// \f$\nabla V_0 \rightarrow V_1\f$. If \f$u_0\f$ is the
/// degree-of-freedom vector associated with \f$V_0\f$, then
/// \f$u_1=Au_0\f$ where \f$u_1\f$ is the degrees-of-freedom vector for
/// interpolating function in the \f$H({\rm curl})\f$ space. An example
/// of where discrete gradient operators are used is the creation of
/// algebraic multigrid solvers for \f$H({\rm curl})\f$ and \f$H({\rm
/// div})\f$ problems.
///
/// @note The sparsity pattern for a discrete operator can be
/// initialised using sparsitybuild::cells. The space `V1` should be
/// used for the rows of the sparsity pattern, `V0` for the columns.
///
/// @warning This function relies on the user supplying appropriate
/// input and output spaces. See parameter descriptions.
///
/// @param[in] topology Mesh topology
/// @param[in] V0 Lagrange element and dofmap for corresponding space to
/// interpolate the gradient from.
/// @param[in] V1 Nédélec (first kind) element and dofmap for
/// corresponding space to interpolate into.
/// @param[in] mat_set A functor that sets values in a matrix
template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
void discrete_gradient(mesh::Topology& topology,
                       std::pair<std::reference_wrapper<const FiniteElement<U>>,
                                 std::reference_wrapper<const DofMap>>
                           V0,
                       std::pair<std::reference_wrapper<const FiniteElement<U>>,
                                 std::reference_wrapper<const DofMap>>
                           V1,
                       auto&& mat_set)
{
  namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;

  auto& e0 = V0.first.get();
  const DofMap& dofmap0 = V0.second.get();
  auto& e1 = V1.first.get();
  const DofMap& dofmap1 = V1.second.get();

  using cmdspan2_t = md::mdspan<const U, md::dextents<std::size_t, 2>>;
  using mdspan2_t = md::mdspan<U, md::dextents<std::size_t, 2>>;
  using cmdspan4_t = md::mdspan<const U, md::dextents<std::size_t, 4>>;

  // Check elements
  if (e0.map_type() != basix::maps::type::identity)
    throw std::runtime_error("Wrong finite element space for V0.");
  if (e0.block_size() != 1)
    throw std::runtime_error("Block size is greater than 1 for V0.");
  if (e0.reference_value_size() != 1)
    throw std::runtime_error("Wrong value size for V0.");

  if (e1.map_type() != basix::maps::type::covariantPiola)
    throw std::runtime_error("Wrong finite element space for V1.");
  if (e1.block_size() != 1)
    throw std::runtime_error("Block size is greater than 1 for V1.");

  // Get V1 (H(curl)) space interpolation points
  const auto [X, Xshape] = e1.interpolation_points();

  // Tabulate first derivatives of Lagrange space at V1 interpolation
  // points
  const int ndofs0 = e0.space_dimension();
  const int tdim = topology.dim();
  std::vector<U> phi0_b((tdim + 1) * Xshape[0] * ndofs0 * 1);
  cmdspan4_t phi0(phi0_b.data(), tdim + 1, Xshape[0], ndofs0, 1);
  e0.tabulate(phi0_b, X, Xshape, 1);

  // Reshape Lagrange basis derivatives as a matrix of shape (tdim *
  // num_points, num_dofs_per_cell)
  cmdspan2_t dphi_reshaped(
      phi0_b.data() + phi0.extent(3) * phi0.extent(2) * phi0.extent(1),
      tdim * phi0.extent(1), phi0.extent(2));

  // Get inverse DOF transform function
  auto apply_inverse_dof_transform = e1.template dof_transformation_fn<T>(
      doftransform::inverse_transpose, false);

  // Generate cell permutations
  topology.create_entity_permutations();
  const std::vector<std::uint32_t>& cell_info
      = topology.get_cell_permutation_info();

  // Create element kernel function

  // Build the element interpolation matrix
  std::vector<T> Ab(e1.space_dimension() * ndofs0);
  {
    md::mdspan<T, md::dextents<std::size_t, 2>> A(Ab.data(),
                                                  e1.space_dimension(), ndofs0);
    const auto [Pi, shape] = e1.interpolation_operator();
    cmdspan2_t _Pi(Pi.data(), shape);
    math::dot(_Pi, dphi_reshaped, A);
  }

  // Insert local interpolation matrix for each cell
  auto cell_map = topology.index_map(tdim);
  assert(cell_map);
  std::int32_t num_cells = cell_map->size_local();
  std::vector<T> Ae(Ab.size());
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    std::ranges::copy(Ab, Ae.begin());
    apply_inverse_dof_transform(Ae, cell_info, c, ndofs0);
    mat_set(dofmap1.cell_dofs(c), dofmap0.cell_dofs(c), Ae);
  }
}

/// @brief Assemble an interpolation operator matrix.
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
/// @param[in] V0 Space to interpolate from.
/// @param[in] V1 Space to interpolate to.
/// @param[in] mat_set Functor that sets values in a matrix.
template <dolfinx::scalar T, std::floating_point U>
void interpolation_matrix(const FunctionSpace<U>& V0,
                          const FunctionSpace<U>& V1, auto&& mat_set)
{
  namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;

  // Get mesh
  auto mesh = V0.mesh();
  assert(mesh);

  // Mesh dims
  const int tdim = mesh->topology()->dim();
  const int gdim = mesh->geometry().dim();

  // Get elements
  std::shared_ptr<const FiniteElement<U>> e0 = V0.element();
  assert(e0);
  std::shared_ptr<const FiniteElement<U>> e1 = V1.element();
  assert(e1);

  std::span<const std::uint32_t> cell_info;
  if (e1->needs_dof_transformations() or e0->needs_dof_transformations())
  {
    mesh->topology_mutable()->create_entity_permutations();
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }

  // Get dofmaps
  auto dofmap0 = V0.dofmap();
  assert(dofmap0);
  auto dofmap1 = V1.dofmap();
  assert(dofmap1);

  // Get block sizes and dof transformation operators
  const int bs0 = e0->block_size();
  const int bs1 = e1->block_size();
  auto apply_dof_transformation0
      = e0->template dof_transformation_fn<U>(doftransform::standard, false);
  auto apply_inverse_dof_transform1 = e1->template dof_transformation_fn<T>(
      doftransform::inverse_transpose, false);

  // Get sizes of elements
  const std::size_t space_dim0 = e0->space_dimension();
  const std::size_t space_dim1 = e1->space_dimension();
  const std::size_t dim0 = space_dim0 / bs0;
  const std::size_t value_size_ref0 = e0->reference_value_size();
  const std::size_t value_size0 = V0.element()->reference_value_size();
  const std::size_t value_size1 = V1.element()->reference_value_size();

  // Get geometry data
  const CoordinateElement<U>& cmap = mesh->geometry().cmap();
  auto x_dofmap = mesh->geometry().dofmap();
  const std::size_t num_dofs_g = cmap.dim();
  std::span<const U> x_g = mesh->geometry().x();

  using mdspan2_t = md::mdspan<U, md::dextents<std::size_t, 2>>;
  using cmdspan2_t = md::mdspan<const U, md::dextents<std::size_t, 2>>;
  using cmdspan3_t = md::mdspan<const U, md::dextents<std::size_t, 3>>;
  using cmdspan4_t = md::mdspan<const U, md::dextents<std::size_t, 4>>;
  using mdspan3_t = md::mdspan<U, md::dextents<std::size_t, 3>>;

  // Evaluate coordinate map basis at reference interpolation points
  const auto [X, Xshape] = e1->interpolation_points();
  std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, Xshape[0]);
  std::vector<U> phi_b(
      std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
  cmdspan4_t phi(phi_b.data(), phi_shape);
  cmap.tabulate(1, X, Xshape, phi_b);

  // Evaluate V0 basis functions at reference interpolation points for V1
  std::vector<U> basis_derivatives_reference0_b(Xshape[0] * dim0
                                                * value_size_ref0);
  cmdspan4_t basis_derivatives_reference0(basis_derivatives_reference0_b.data(),
                                          1, Xshape[0], dim0, value_size_ref0);
  e0->tabulate(basis_derivatives_reference0_b, X, Xshape, 0);

  // Clamp values
  std::ranges::transform(
      basis_derivatives_reference0_b, basis_derivatives_reference0_b.begin(),
      [atol = 1e-14](auto x) { return std::abs(x) < atol ? 0.0 : x; });

  // Create working arrays
  std::vector<U> basis_reference0_b(Xshape[0] * dim0 * value_size_ref0);
  mdspan3_t basis_reference0(basis_reference0_b.data(), Xshape[0], dim0,
                             value_size_ref0);
  std::vector<U> J_b(Xshape[0] * gdim * tdim);
  mdspan3_t J(J_b.data(), Xshape[0], gdim, tdim);
  std::vector<U> K_b(Xshape[0] * tdim * gdim);
  mdspan3_t K(K_b.data(), Xshape[0], tdim, gdim);
  std::vector<U> detJ(Xshape[0]);
  std::vector<U> det_scratch(2 * tdim * gdim);

  // Get the interpolation operator (matrix) `Pi` that maps a function
  // evaluated at the interpolation points to the element degrees of
  // freedom, i.e. dofs = Pi f_x
  const auto [_Pi_1, pi_shape] = e1->interpolation_operator();
  cmdspan2_t Pi_1(_Pi_1.data(), pi_shape);

  bool interpolation_ident = e1->interpolation_ident();

  using u_t = md::mdspan<U, md::dextents<std::size_t, 2>>;
  using U_t = md::mdspan<const U, md::dextents<std::size_t, 2>>;
  using J_t = md::mdspan<const U, md::dextents<std::size_t, 2>>;
  using K_t = md::mdspan<const U, md::dextents<std::size_t, 2>>;
  auto push_forward_fn0
      = e0->basix_element().template map_fn<u_t, U_t, J_t, K_t>();

  // Basis values of Lagrange space unrolled for block size
  // (num_quadrature_points, Lagrange dof, value_size)
  std::vector<U> basis_values_b(Xshape[0] * bs0 * dim0
                                * V1.element()->value_size());
  mdspan3_t basis_values(basis_values_b.data(), Xshape[0], bs0 * dim0,
                         V1.element()->value_size());
  std::vector<U> mapped_values_b(Xshape[0] * bs0 * dim0
                                 * V1.element()->value_size());
  mdspan3_t mapped_values(mapped_values_b.data(), Xshape[0], bs0 * dim0,
                          V1.element()->value_size());

  auto pull_back_fn1
      = e1->basix_element().template map_fn<u_t, U_t, K_t, J_t>();

  std::vector<U> coord_dofs_b(num_dofs_g * gdim);
  mdspan2_t coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);
  std::vector<U> basis0_b(Xshape[0] * dim0 * value_size0);
  mdspan3_t basis0(basis0_b.data(), Xshape[0], dim0, value_size0);

  // Buffers
  std::vector<T> Ab(space_dim0 * space_dim1);
  std::vector<T> local1(space_dim1);

  // Iterate over mesh and interpolate on each cell
  auto cell_map = mesh->topology()->index_map(tdim);
  assert(cell_map);
  std::int32_t num_cells = cell_map->size_local();
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    // Get cell geometry (coordinate dofs)
    auto x_dofs = md::submdspan(x_dofmap, c, md::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      for (std::size_t j = 0; j < gdim; ++j)
        coord_dofs(i, j) = x_g[3 * x_dofs[i] + j];
    }

    // Compute Jacobians and reference points for current cell
    std::ranges::fill(J_b, 0);
    for (std::size_t p = 0; p < Xshape[0]; ++p)
    {
      auto dphi
          = md::submdspan(phi, std::pair(1, tdim + 1), p, md::full_extent, 0);
      auto _J = md::submdspan(J, p, md::full_extent, md::full_extent);
      cmap.compute_jacobian(dphi, coord_dofs, _J);
      auto _K = md::submdspan(K, p, md::full_extent, md::full_extent);
      cmap.compute_jacobian_inverse(_J, _K);
      detJ[p] = cmap.compute_jacobian_determinant(_J, det_scratch);
    }

    // Copy evaluated basis on reference, apply DOF transformations, and
    // push forward to physical element
    for (std::size_t k0 = 0; k0 < basis_reference0.extent(0); ++k0)
      for (std::size_t k1 = 0; k1 < basis_reference0.extent(1); ++k1)
        for (std::size_t k2 = 0; k2 < basis_reference0.extent(2); ++k2)
          basis_reference0(k0, k1, k2)
              = basis_derivatives_reference0(0, k0, k1, k2);
    for (std::size_t p = 0; p < Xshape[0]; ++p)
    {
      apply_dof_transformation0(
          std::span(basis_reference0.data_handle() + p * dim0 * value_size_ref0,
                    dim0 * value_size_ref0),
          cell_info, c, value_size_ref0);
    }

    for (std::size_t p = 0; p < basis0.extent(0); ++p)
    {
      auto _u = md::submdspan(basis0, p, md::full_extent, md::full_extent);
      auto _U = md::submdspan(basis_reference0, p, md::full_extent,
                              md::full_extent);
      auto _K = md::submdspan(K, p, md::full_extent, md::full_extent);
      auto _J = md::submdspan(J, p, md::full_extent, md::full_extent);
      push_forward_fn0(_u, _U, _J, detJ[p], _K);
    }

    // Unroll basis function for input space for block size
    for (std::size_t p = 0; p < Xshape[0]; ++p)
      for (std::size_t i = 0; i < dim0; ++i)
        for (std::size_t j = 0; j < value_size0; ++j)
          for (int k = 0; k < bs0; ++k)
            basis_values(p, i * bs0 + k, j * bs0 + k) = basis0(p, i, j);

    // Pull back the physical values to the reference of output space
    for (std::size_t p = 0; p < basis_values.extent(0); ++p)
    {
      auto _u
          = md::submdspan(basis_values, p, md::full_extent, md::full_extent);
      auto _U
          = md::submdspan(mapped_values, p, md::full_extent, md::full_extent);
      auto _K = md::submdspan(K, p, md::full_extent, md::full_extent);
      auto _J = md::submdspan(J, p, md::full_extent, md::full_extent);
      pull_back_fn1(_U, _u, _K, 1.0 / detJ[p], _J);
    }

    // Apply interpolation matrix to basis values of V0 at the
    // interpolation points of V1
    if (interpolation_ident)
    {
      md::mdspan<T, md::dextents<std::size_t, 3>> A(
          Ab.data(), Xshape[0], V1.element()->value_size(), space_dim0);
      for (std::size_t i = 0; i < mapped_values.extent(0); ++i)
        for (std::size_t j = 0; j < mapped_values.extent(1); ++j)
          for (std::size_t k = 0; k < mapped_values.extent(2); ++k)
            A(i, k, j) = mapped_values(i, j, k);
    }
    else
    {
      for (std::size_t i = 0; i < mapped_values.extent(1); ++i)
      {
        auto values
            = md::submdspan(mapped_values, md::full_extent, i, md::full_extent);
        impl::interpolation_apply(Pi_1, values, std::span(local1), bs1);
        for (std::size_t j = 0; j < local1.size(); j++)
          Ab[space_dim0 * j + i] = local1[j];
      }
    }

    apply_inverse_dof_transform1(Ab, cell_info, c, space_dim0);
    mat_set(dofmap1->cell_dofs(c), dofmap0->cell_dofs(c), Ab);
  }
}

} // namespace dolfinx::fem
