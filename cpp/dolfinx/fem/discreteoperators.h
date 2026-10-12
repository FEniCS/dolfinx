// Copyright (C) 2015-2026 Garth N. Wells, Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CoordinateElement.h"
#include "DofMap.h"
#include "ElementDofLayout.h"
#include "FiniteElement.h"
#include "FunctionSpace.h"
#include "interpolate.h"
#include "sparsitybuild.h"
#include <algorithm>
#include <array>
#include <basix/sobolev-spaces.h>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/math.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <exception>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dolfinx::fem
{

namespace impl
{
/// @brief Check source conformity for an entity-closure derivative stencil.
/// @param[in] element Source element.
template <std::floating_point T>
void check_derivative_source(const FiniteElement<T>& element)
{
  const basix::sobolev::space space = element.basix_element().sobolev_space();
  const bool h1
      = basix::sobolev::space_intersection(space, basix::sobolev::space::H1)
        == basix::sobolev::space::H1;
  if (element.map_type() == basix::maps::type::identity and !h1)
  {
    throw std::invalid_argument(
        "Source element must be H1-conforming for the discrete gradient.");
  }
  if (element.map_type() == basix::maps::type::covariantPiola and !h1
      and space != basix::sobolev::space::HCurl)
  {
    throw std::invalid_argument(
        "Source element must be H(curl)-conforming for the discrete curl.");
  }
}

/// @brief Gather and insert the entity-closure blocks of one cell's
/// element matrix.
///
/// @param[in] blocks Entity blocks, from
/// sparsitybuild::entity_closure_blocks.
/// @param[in] A Element matrix, its rows indexed by the space of
/// `cell_rows`.
/// @param[in] ncols Row stride of `A`.
/// @param[in] cell_rows Row-space degrees-of-freedom of the cell.
/// @param[in] cell_cols Column-space degrees-of-freedom of the cell.
/// @param[out] rows,cols,Ae Scratch, each at least as large as the
/// element matrix dimension it indexes.
/// @param[in] mat_set Functor that sets values in a matrix.
template <dolfinx::scalar T>
void insert_entity_blocks(
    std::span<const std::pair<std::span<const int>, std::span<const int>>>
        blocks,
    std::span<const T> A, std::size_t ncols,
    std::span<const std::int32_t> cell_rows,
    std::span<const std::int32_t> cell_cols, std::span<std::int32_t> rows,
    std::span<std::int32_t> cols, std::span<T> Ae, auto&& mat_set)
{
  for (const auto& [rdofs, cdofs] : blocks)
  {
    std::ranges::transform(cdofs, cols.begin(),
                           [cell_cols](int d) { return cell_cols[d]; });
    for (std::size_t i = 0; i < rdofs.size(); ++i)
    {
      rows[i] = cell_rows[rdofs[i]];
      for (std::size_t j = 0; j < cdofs.size(); ++j)
        Ae[i * cdofs.size() + j] = A[rdofs[i] * ncols + cdofs[j]];
    }
    mat_set(rows.first(rdofs.size()), cols.first(cdofs.size()),
            Ae.first(rdofs.size() * cdofs.size()));
  }
}

} // namespace impl

/// @brief Assemble a discrete curl operator.
///
/// For vector-valued finite functions \f$u \in V_{0} \f$ and \f$v \in
/// V_{1}\f$, consider the interpolation of the curl of \f$u\f$ in the
/// space \f$V_{1}\f$, i.e. \f$\Pi_{V_{1}}: \nabla \times u \rightarrow
/// v\f$, where \f$\Pi_{V_{1}}\f$ is the interpolation operator
/// associated with \f$V_{1}\f$. This interpolation of \f$\nabla \times
/// u\f$ into \f$V_{1}\f$ is properly posed and exact for specific
/// choices of function spaces. If \f$V_{0}\f$ is a Nédélec (\f$H({\rm
/// curl})\f$) space of degree \f$k > 1\f$ and \f$V_{1}\f$ is a
/// Raviart-Thomas (\f$H({\rm div})\f$) space of degree of at least \f$k
/// - 1\f$, then the interpolation is exact.
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
/// Guermond (2021), *Finite Elements I*, Springer Nature,
/// https://doi.org/10.1007/978-3-030-56341-7 [Corollary 9.9 (Commuting
/// properties)]. Consequently, the spaces `V0` and `V1` must use
/// covariant and contravariant maps, respectively.
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
/// multigrid preconditioners for \f$H({\rm div})\f$ problems, e.g. when
/// using the Hypre Auxiliary-space Divergence Solver (ADS) to solve a
/// mixed Poisson in three-dimension.
///
/// @pre `V0` and `V1` must be vector-valued, in three spatial
/// dimensions, and use covariant and contravariant maps, respectively.
/// `V0` must be H(curl)-conforming.
///
/// @note Values are inserted one mesh entity at a time: the `V1`
/// degrees-of-freedom on an entity against the `V0` degrees-of-freedom
/// on its closure. A facet degree-of-freedom of `V1` is a moment of
/// \f$n \cdot \nabla \times u\f$ over the facet, which by Stokes'
/// theorem is a circulation of \f$u\f$ around its boundary, so it sees
/// only `V0` on the closure of that facet. Build the sparsity pattern
/// with sparsitybuild::entity_closure, which holds exactly those
/// couplings. sparsitybuild::cells is wide enough too, at the cost of
/// entries the operator cannot occupy.
///
/// @tparam T Scalar type of the mesh and elements. @tparam U Scalar
/// type of the matrix being inserted into. This is usually the same as
/// `T`, but may differ for matrix backends that support only a specific
/// type, e.g. PETSc which supports only one scalar type for a build of
/// PETSc.
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
  // Get mesh
  auto mesh = V0.mesh();
  assert(mesh);
  assert(V1.mesh());
  if (mesh != V1.mesh())
    throw std::invalid_argument("Meshes must be the same.");

  if (mesh->geometry().dim() != 3)
    throw std::invalid_argument("Geometric must be equal to 3..");
  if (mesh->geometry().dim() != mesh->topology()->dim())
  {
    throw std::invalid_argument(
        "Geometric and topological dimensions must be equal.");
  }
  constexpr int gdim = 3;

  // Get elements
  std::shared_ptr<const FiniteElement<T>> e0 = V0.element();
  assert(e0);
  if (e0->map_type() != basix::maps::type::covariantPiola)
  {
    throw std::invalid_argument(
        "Finite element for parent space must be covariant Piola.");
  }
  impl::check_derivative_source(*e0);

  std::shared_ptr<const FiniteElement<T>> e1 = V1.element();
  assert(e1);
  if (e1->map_type() != basix::maps::type::contravariantPiola)
  {
    throw std::invalid_argument(
        "Finite element for target space must be contracovariant Piola.");
  }

  // Get cell orientation information
  std::span<const std::uint32_t> cell_info;
  if (e1->needs_dof_transformations() or e0->needs_dof_transformations())
  {
    mesh->topology_mutable()->create_cell_permutations();
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
    throw std::invalid_argument("Value size for parent space should be 3.");
  if (e1->reference_value_size() != 3)
    throw std::invalid_argument("Value size for target space should be 3.");

  // Get the V1 reference interpolation points
  const auto [X, Xshape] = e1->interpolation_points();

  // Evaluate V0 basis function derivatives at reference interpolation
  // points for V1, (deriv, pt_idx, phi (dof), comp)
  const auto [Phi0_b, Phi0_shape] = e0->tabulate(X, Xshape, 1);
  md::mdspan<const T, md::extents<std::size_t, 4, md::dynamic_extent,
                                  md::dynamic_extent, 3>>
      Phi0(Phi0_b.data(), Phi0_shape);

  // Create working arrays, (point, phi (dof), deriv, comp)
  md::extents<std::size_t, md::dynamic_extent, md::dynamic_extent, 3, 3>
      dPhi0_ext(Phi0.extent(1), Phi0.extent(2), Phi0.extent(0) - 1,
                Phi0.extent(3));
  std::vector<T> dPhi0_b(dPhi0_ext.extent(0) * dPhi0_ext.extent(1)
                         * dPhi0_ext.extent(2) * dPhi0_ext.extent(3));
  md::mdspan<T, decltype(dPhi0_ext)> dPhi0(dPhi0_b.data(), dPhi0_ext);

  // Get the interpolation operator (matrix) Pi that maps a function
  // evaluated at the interpolation points to the V1 element degrees of
  // freedom, i.e. dofs = Pi f_x
  const auto [Pi1_b, pi_shape] = e1->interpolation_operator();
  md::mdspan<const T, md::dextents<std::size_t, 2>> Pi_1(Pi1_b.data(),
                                                         pi_shape);

  // curl data structure, (pt_idx, phi (dof), comp)
  std::vector<T> curl_b(dPhi0.extent(0) * dPhi0.extent(1) * dPhi0.extent(3));
  md::mdspan<
      T, md::extents<std::size_t, md::dynamic_extent, md::dynamic_extent, 3>>
      curl(curl_b.data(), dPhi0.extent(0), dPhi0.extent(1), dPhi0.extent(3));

  std::vector<U> Ab(space_dim0 * space_dim1);

  // Insert the element matrix one mesh entity at a time, so that the
  // operator gets its exact sparsity
  const std::vector<std::pair<std::span<const int>, std::span<const int>>>
      blocks = sparsitybuild::entity_closure_blocks(
          dofmap1->element_dof_layout(), dofmap0->element_dof_layout());
  std::vector<U> Ab_e(Ab.size());
  std::vector<std::int32_t> rows(space_dim1), cols(space_dim0);

  // Iterate over mesh and interpolate on each cell
  assert(mesh->topology()->index_map(gdim));
  for (std::int32_t c = 0; c < mesh->topology()->index_map(gdim)->size_local();
       ++c)
  {
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
      if (apply_dof_transformation0)
      {
        apply_dof_transformation0(std::span(dPhi0.data_handle() + offset, size),
                                  cell_info, c,
                                  dPhi0.extent(2) * dPhi0.extent(3));
      }
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

    // Apply interpolation matrix to basis derivative values of V0 at
    // the interpolation points of V1. Pi_1 does not depend on the
    // basis function index i, so all space_dim0 applications are
    // combined into a single loop nest (rather than calling
    // interpolation_apply once per basis function) so that each row
    // of Pi_1 is read from memory once instead of space_dim0 times.
    std::ranges::fill(Ab, U(0));
    for (std::size_t j = 0; j < space_dim1; ++j)         // V1 dof
      for (std::size_t k = 0; k < curl.extent(2); ++k)   // component
        for (std::size_t p = 0; p < curl.extent(0); ++p) // point
        {
          T pi_val = Pi_1(j, k * curl.extent(0) + p);
          for (std::size_t i = 0; i < space_dim0; ++i) // V0 basis function
            Ab[space_dim0 * j + i] += static_cast<U>(pi_val * curl(p, i, k));
        }

    if (apply_inverse_dof_transform1)
      apply_inverse_dof_transform1(Ab, cell_info, c, space_dim0);

    impl::insert_entity_blocks<U>(blocks, Ab, space_dim0, dofmap1->cell_dofs(c),
                                  dofmap0->cell_dofs(c), rows, cols, Ab_e,
                                  mat_set);
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
/// @note Values are inserted one mesh entity at a time: the `V1`
/// degrees-of-freedom on an entity against the `V0` degrees-of-freedom
/// on its closure. Build the sparsity pattern with
/// sparsitybuild::entity_closure, which holds exactly those couplings.
/// sparsitybuild::cells is wide enough too, at the cost of entries the
/// operator cannot occupy.
///
/// @warning This function relies on the user supplying appropriate
/// input and output spaces. See parameter descriptions.
///
/// @param[in] topology Mesh topology
/// @param[in] V0 Lagrange element and dofmap for corresponding space to
/// interpolate the gradient from. The element must be H1-conforming.
/// @param[in] V1 Nédélec (first kind) element and dofmap for
/// corresponding space to interpolate into.
/// @param[in] mat_set A functor that sets values in a matrix
template <dolfinx::scalar T, std::floating_point U = dolfinx::scalar_value_t<T>>
void discrete_gradient(mesh::Topology& topology,
                       std::pair<std::reference_wrapper<const FiniteElement<U>>,
                                 std::reference_wrapper<const DofMap>>
                           V0,
                       std::pair<std::reference_wrapper<const FiniteElement<U>>,
                                 std::reference_wrapper<const DofMap>>
                           V1,
                       auto&& mat_set)
{
  auto& e0 = V0.first.get();
  const DofMap& dofmap0 = V0.second.get();
  auto& e1 = V1.first.get();
  const DofMap& dofmap1 = V1.second.get();

  using cmdspan2_t = md::mdspan<const U, md::dextents<std::size_t, 2>>;
  using cmdspan4_t = md::mdspan<const U, md::dextents<std::size_t, 4>>;

  // Check elements
  if (e0.map_type() != basix::maps::type::identity)
    throw std::invalid_argument("Wrong finite element space for V0.");
  if (e0.block_size() != 1)
    throw std::invalid_argument("Block size is greater than 1 for V0.");
  if (e0.reference_value_size() != 1)
    throw std::invalid_argument("Wrong value size for V0.");
  impl::check_derivative_source(e0);

  if (e1.map_type() != basix::maps::type::covariantPiola)
    throw std::invalid_argument("Wrong finite element space for V1.");
  if (e1.block_size() != 1)
    throw std::invalid_argument("Block size is greater than 1 for V1.");

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
  topology.create_cell_permutations();
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

  // Insert the element matrix one mesh entity at a time, so that the
  // operator gets its exact sparsity
  const std::vector<std::pair<std::span<const int>, std::span<const int>>>
      blocks = sparsitybuild::entity_closure_blocks(
          dofmap1.element_dof_layout(), dofmap0.element_dof_layout());
  auto cell_map = topology.index_map(tdim);
  assert(cell_map);
  std::int32_t num_cells = cell_map->size_local();
  std::vector<T> Ae(Ab.size()), Ab_e(Ab.size());
  std::vector<std::int32_t> rows(e1.space_dimension()), cols(ndofs0);
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    std::ranges::copy(Ab, Ae.begin());
    if (apply_inverse_dof_transform)
      apply_inverse_dof_transform(Ae, cell_info, c, ndofs0);
    impl::insert_entity_blocks<T>(blocks, Ae, ndofs0, dofmap1.cell_dofs(c),
                                  dofmap0.cell_dofs(c), rows, cols, Ab_e,
                                  mat_set);
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
/// Unlike discrete_gradient and discrete_curl there is no narrower
/// pattern: `V0` is general here, so its restriction to an entity need
/// not be determined by the degrees-of-freedom on that entity's
/// closure, and sparsitybuild::entity_closure does not apply.
///
/// @param[in] V0 Space to interpolate from.
/// @param[in] V1 Space to interpolate to.
/// @param[in] mat_add Functor that adds values to a matrix.
template <dolfinx::scalar T, std::floating_point U>
void interpolation_matrix(const FunctionSpace<U>& V0,
                          const FunctionSpace<U>& V1, auto&& mat_add)
{
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

  if (!std::ranges::equal(e0->value_shape(), e1->value_shape()))
  {
    throw std::invalid_argument(
        "Interpolation operator: elements have different value dimensions");
  }

  std::span<const std::uint32_t> cell_info;
  if (e1->needs_dof_transformations() or e0->needs_dof_transformations())
  {
    mesh->topology_mutable()->create_cell_permutations();
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
  // basis0 holds one block of V0's basis pushed forward to the physical
  // cell; basis_values holds all bs0 blocks; mapped_values holds them
  // pulled back to the reference cell of e1.
  const std::size_t value_size0 = e0->physical_base_value_size();
  const std::size_t value_size_phys = e0->value_size();
  const std::size_t value_size_ref1
      = e1->reference_value_size() * static_cast<std::size_t>(bs1);

  // Get geometry data
  const CoordinateElement<U>& cmap = mesh->geometry().cmaps().front();
  auto x_dofmap = mesh->geometry().dofmaps().front();
  const std::size_t num_dofs_g = cmap.dim();
  std::span<const U> x_g = mesh->geometry().x();

  using mdspan2_t = md::mdspan<U, md::dextents<std::size_t, 2>>;
  using cmdspan2_t = md::mdspan<const U, md::dextents<std::size_t, 2>>;
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

  // Select how Pi_1 is applied once, outside the cell loop
  auto apply_Pi_1 = e1->interpolation_ident()
                        ? impl::interpolation_apply_columns<true, T, U>
                        : impl::interpolation_apply_columns<false, T, U>;

  using u_t = md::mdspan<U, md::dextents<std::size_t, 2>>;
  using U_t = md::mdspan<const U, md::dextents<std::size_t, 2>>;
  using J_t = md::mdspan<const U, md::dextents<std::size_t, 2>>;
  using K_t = md::mdspan<const U, md::dextents<std::size_t, 2>>;
  auto push_forward_fn0
      = e0->basix_element().template map_fn<u_t, U_t, J_t, K_t>();

  // Basis values of Lagrange space unrolled for block size
  // (num_quadrature_points, Lagrange dof, value_size)
  std::vector<U> basis_values_b(Xshape[0] * bs0 * dim0 * value_size_phys);
  mdspan3_t basis_values(basis_values_b.data(), Xshape[0], bs0 * dim0,
                         value_size_phys);
  std::vector<U> mapped_values_b(Xshape[0] * bs0 * dim0 * value_size_ref1);
  mdspan3_t mapped_values(mapped_values_b.data(), Xshape[0], bs0 * dim0,
                          value_size_ref1);

  auto pull_back_fn1
      = e1->basix_element().template map_fn<u_t, U_t, K_t, J_t>();

  std::vector<U> coord_dofs_b(num_dofs_g * gdim);
  mdspan2_t coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);
  std::vector<U> basis0_b(Xshape[0] * dim0 * value_size0);
  mdspan3_t basis0(basis0_b.data(), Xshape[0], dim0, value_size0);

  // Buffers
  std::vector<T> Ab(space_dim0 * space_dim1);

  // Iterate over mesh and interpolate on each cell
  auto cell_map = mesh->topology()->index_map(tdim);
  assert(cell_map);
  std::int32_t num_cells = cell_map->size_local();
  int row_bs = dofmap1->bs();
  std::int32_t num_owned_rows
      = dofmap1->index_map->size_local() * dofmap1->index_map_bs() / row_bs;
  std::int32_t num_ghosted_rows
      = dofmap1->index_map->num_ghosts() * dofmap1->index_map_bs() / row_bs;
  std::vector<std::int8_t> row_added(num_owned_rows + num_ghosted_rows, 0);

  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    // Get cell geometry (coordinate dofs)
    auto x_dofs = md::submdspan(x_dofmap, c, md::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      for (int j = 0; j < gdim; ++j)
        coord_dofs(i, j) = x_g[3 * x_dofs[i] + j];
    }

    // Compute Jacobians at the interpolation points of V1
    impl::compute_jacobians<U>(cmap, phi, coord_dofs, J, K, detJ, det_scratch);

    // Copy evaluated basis on reference (a fresh copy is needed each
    // cell as DOF transformations below are applied in place), and
    // push forward to physical element. The source and destination
    // have identical memory layout (the leading, unit-length
    // derivative axis of basis_derivatives_reference0 is a no-op for
    // indexing purposes), so a flat copy is used instead of an
    // element-by-element mdspan loop.
    std::ranges::copy(basis_derivatives_reference0_b,
                      basis_reference0_b.begin());
    if (apply_dof_transformation0)
    {
      for (std::size_t p = 0; p < Xshape[0]; ++p)
      {
        apply_dof_transformation0(std::span(basis_reference0.data_handle()
                                                + p * dim0 * value_size_ref0,
                                            dim0 * value_size_ref0),
                                  cell_info, c, value_size_ref0);
      }
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
    apply_Pi_1(Pi_1, mapped_values, bs1, Ab);

    if (apply_inverse_dof_transform1)
      apply_inverse_dof_transform1(Ab, cell_info, c, space_dim0);

    // Zero out all rows that have already been added on this process
    // and only add owned rows
    {
      md::mdspan<T, md::dextents<std::size_t, 2>> A(Ab.data(), space_dim1,
                                                    space_dim0);
      auto row_dofs = dofmap1->cell_dofs(c);
      for (std::size_t i = 0; i < row_dofs.size(); ++i)
      {
        std::int32_t r = row_dofs[i];
        if (r >= num_owned_rows || row_added[r])
        {
          for (std::size_t j = 0; j < space_dim0; ++j)
            for (int k = 0; k < row_bs; ++k)
              A(i * row_bs + k, j) = 0.0;
        }
        row_added[r] = 1;
      }
    }
    mat_add(dofmap1->cell_dofs(c), dofmap0->cell_dofs(c), Ab);
  }
}

/// @brief Basis functions of a function space evaluated at points that
/// another process requested.
///
/// Created by evaluate_basis_nonmatching. Rows follow the order of the
/// points that this process sent into the ownership determination.
template <std::floating_point T>
struct NonmatchingBasisData
{
  /// Basis values of one block of the element, pushed forward,
  /// `shape=(num_points, num_cell_dofs, physical_base_value_size)`. Zero
  /// for a point that no process owns.
  std::vector<T> values;

  /// Global (blocked) degree-of-freedom indices of the cell containing
  /// each point, `shape=(num_points, num_cell_dofs)`. `-1` for a point
  /// that no process owns.
  std::vector<std::int64_t> dofs;

  /// Owning process of each entry of `dofs`.
  std::vector<int> owners;
};

/// @brief Evaluate the basis functions of a space at points owned by
/// other processes, and return them to the processes that requested
/// them.
///
/// Each process evaluates the basis at the points it owns
/// (`interpolation_data.dest_points`) and sends the values, together
/// with the global indices and owners of the degrees-of-freedom of the
/// containing cells, back to the process that requested each point.
///
/// @note Collective. A failure to pull back a point (non-affine
/// geometry) is raised on every process, after the communication.
///
/// @param[in] V0 Space whose basis is evaluated.
/// @param[in] interpolation_data Ownership of the points, e.g. from
/// geometry::determine_point_ownership on the mesh of `V0`.
/// @param[in] tol Tolerance of the Newton method for non-affine
/// pull-backs.
/// @param[in] maxit Maximum number of Newton iterations for non-affine
/// pull-backs.
/// @param[in] num_threads Number of threads used for the evaluation.
/// @return Basis values and degree-of-freedom data at the points this
/// process requested.
template <std::floating_point T>
NonmatchingBasisData<T> evaluate_basis_nonmatching(
    const FunctionSpace<T>& V0,
    const geometry::PointOwnershipData<T>& interpolation_data, double tol,
    int maxit, int num_threads = 1)
{
  auto mesh = V0.mesh();
  assert(mesh);
  std::shared_ptr<const FiniteElement<T>> element = V0.element();
  assert(element);
  std::shared_ptr<const DofMap> dofmap = V0.dofmap();
  assert(dofmap);

  if (int num_sub_elements = element->num_sub_elements();
      num_sub_elements > 1 and num_sub_elements != element->block_size())
  {
    throw std::invalid_argument("Evaluation of basis functions is not "
                                "supported for mixed elements. Extract "
                                "subspaces.");
  }

  const std::size_t num_cell_dofs = dofmap->map().extent(1);
  assert(num_cell_dofs
         == static_cast<std::size_t>(element->space_dimension()
                                     / element->block_size()));
  const std::size_t value_size = element->physical_base_value_size();
  const std::size_t block = num_cell_dofs * value_size;

  // Evaluate the basis at the points owned by this process. A failure
  // is held back until all processes have communicated.
  std::span<const std::int32_t> cells = interpolation_data.dest_cells;
  const std::size_t num_owned = cells.size();
  std::vector<T> values(num_owned * block, 0);
  std::exception_ptr error;
  try
  {
    impl::evaluate_basis<T>(*element, *mesh, interpolation_data.dest_points,
                            {num_owned, 3}, cells, tol, maxit, num_threads,
                            [&values, block](std::size_t p, std::int32_t,
                                             impl::mdspan_t<const T, 2> basis)
                            {
                              std::copy_n(basis.data_handle(), block,
                                          std::next(values.begin(), p * block));
                            });
  }
  catch (...)
  {
    error = std::current_exception();
  }

  // Global indices and owners of the degrees-of-freedom of each cell
  MPI_Comm comm = mesh->comm();
  const int rank = dolfinx::MPI::rank(comm);
  auto index_map = dofmap->index_map;
  assert(index_map);
  const std::int32_t size_local = index_map->size_local();
  std::span<const int> ghost_owners = index_map->owners();
  std::vector<std::int64_t> dofs(num_owned * num_cell_dofs);
  std::vector<int> owners(num_owned * num_cell_dofs);
  for (std::size_t p = 0; p < num_owned; ++p)
  {
    std::span<const std::int32_t> cell_dofs = dofmap->cell_dofs(cells[p]);
    index_map->local_to_global(
        cell_dofs, std::span(dofs).subspan(p * num_cell_dofs, num_cell_dofs));
    for (std::size_t i = 0; i < num_cell_dofs; ++i)
    {
      std::int32_t dof = cell_dofs[i];
      owners[p * num_cell_dofs + i]
          = dof < size_local ? rank : ghost_owners[dof - size_local];
    }
  }

  // Send to the processes that requested the points
  std::span<const int> src_ranks = interpolation_data.dest_owners;
  std::span<const int> dest_ranks = interpolation_data.src_owner;
  const std::size_t num_points = dest_ranks.size();
  NonmatchingBasisData<T> data{
      std::vector<T>(num_points * block),
      std::vector<std::int64_t>(num_points * num_cell_dofs),
      std::vector<int>(num_points * num_cell_dofs)};
  impl::scatter_values(
      comm, src_ranks, dest_ranks,
      impl::mdspan_t<const T, 2>(values.data(), num_owned, block),
      std::span(data.values));
  impl::scatter_values(comm, src_ranks, dest_ranks,
                       impl::mdspan_t<const std::int64_t, 2>(
                           dofs.data(), num_owned, num_cell_dofs),
                       std::span(data.dofs));
  impl::scatter_values(
      comm, src_ranks, dest_ranks,
      impl::mdspan_t<const int, 2>(owners.data(), num_owned, num_cell_dofs),
      std::span(data.owners));
  for (std::size_t p = 0; p < num_points; ++p)
  {
    if (dest_ranks[p] < 0)
    {
      std::fill_n(std::next(data.dofs.begin(), p * num_cell_dofs),
                  num_cell_dofs, -1);
    }
  }

  // Raise a failure on every process
  int failed = error ? 1 : 0;
  int any_failed = 0;
  MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, comm);
  if (error)
    std::rethrow_exception(error);
  else if (any_failed)
  {
    throw std::runtime_error(
        "Evaluation of basis functions failed on another process.");
  }

  return data;
}

/// @brief Create the column index map of a nonmatching interpolation
/// matrix.
///
/// This is the index map of `V0` with the degrees-of-freedom of `data`
/// that are neither owned nor ghosts on this process appended as
/// ghosts. The owned range and the existing ghosts, in their order, are
/// those of `V0`.
///
/// @note Collective.
///
/// @param[in] V0 Space interpolated from.
/// @param[in] data Basis data from evaluate_basis_nonmatching.
/// @return Index map for the columns.
template <std::floating_point T>
common::IndexMap create_column_map(const FunctionSpace<T>& V0,
                                   const NonmatchingBasisData<T>& data)
{
  auto index_map = V0.dofmap()->index_map;
  assert(index_map);
  const std::array<std::int64_t, 2> range = index_map->local_range();
  std::span<const std::int64_t> ghosts0 = index_map->ghosts();
  std::span<const int> owners0 = index_map->owners();

  std::vector<std::int64_t> ghosts0_sorted(ghosts0.begin(), ghosts0.end());
  std::ranges::sort(ghosts0_sorted);

  // Degrees-of-freedom that are neither owned nor ghosts
  std::vector<std::pair<std::int64_t, int>> new_ghosts;
  for (std::size_t i = 0; i < data.dofs.size(); ++i)
  {
    std::int64_t dof = data.dofs[i];
    if (dof < 0 or (dof >= range[0] and dof < range[1])
        or std::ranges::binary_search(ghosts0_sorted, dof))
    {
      continue;
    }
    new_ghosts.emplace_back(dof, data.owners[i]);
  }
  std::ranges::sort(new_ghosts);
  new_ghosts.erase(std::ranges::unique(new_ghosts).begin(), new_ghosts.end());

  std::vector<std::int64_t> ghosts(ghosts0.begin(), ghosts0.end());
  std::vector<int> owners(owners0.begin(), owners0.end());
  for (auto [dof, owner] : new_ghosts)
  {
    ghosts.push_back(dof);
    owners.push_back(owner);
  }

  return common::IndexMap(index_map->comm(), index_map->size_local(), ghosts,
                          owners);
}

namespace impl
{
/// @brief Column indices, local to `column_map`, of the
/// degrees-of-freedom in `data`, with `-1` for unowned points.
template <std::floating_point T>
std::vector<std::int32_t>
nonmatching_columns(const NonmatchingBasisData<T>& data,
                    const common::IndexMap& column_map)
{
  std::vector<std::int64_t> global;
  global.reserve(data.dofs.size());
  std::ranges::copy_if(data.dofs, std::back_inserter(global),
                       [](std::int64_t dof) { return dof >= 0; });
  std::vector<std::int32_t> local(global.size());
  column_map.global_to_local(global, local);

  std::vector<std::int32_t> columns(data.dofs.size(), -1);
  for (std::size_t i = 0, j = 0; i < data.dofs.size(); ++i)
  {
    if (data.dofs[i] >= 0)
      columns[i] = local[j++];
  }
  return columns;
}

/// @brief Sorted, unique columns of the points of one cell, ignoring
/// `-1`.
inline void cell_columns(std::span<const std::int32_t> point_columns,
                         std::vector<std::int32_t>& columns)
{
  columns.clear();
  std::ranges::copy_if(point_columns, std::back_inserter(columns),
                       [](std::int32_t col) { return col >= 0; });
  std::ranges::sort(columns);
  columns.erase(std::ranges::unique(columns).begin(), columns.end());
}
} // namespace impl

/// @brief Insert the non-zero pattern of a nonmatching interpolation
/// matrix into a sparsity pattern.
///
/// @param[in,out] pattern Sparsity pattern with the index map of `V1`
/// for the rows and `column_map` for the columns.
/// @param[in] V1 Space interpolated to.
/// @param[in] cells Cells of the mesh of `V1` whose interpolation
/// points were used to create `data`.
/// @param[in] data Basis data from evaluate_basis_nonmatching.
/// @param[in] column_map Column index map from create_column_map.
template <std::floating_point T>
void sparsity_nonmatching(la::SparsityPattern& pattern,
                          const FunctionSpace<T>& V1,
                          std::span<const std::int32_t> cells,
                          const NonmatchingBasisData<T>& data,
                          const common::IndexMap& column_map)
{
  if (cells.empty())
    return;

  std::shared_ptr<const DofMap> dofmap1 = V1.dofmap();
  assert(dofmap1);
  const std::vector<std::int32_t> columns
      = impl::nonmatching_columns(data, column_map);
  const std::size_t points_per_cell = columns.size() / cells.size();
  std::vector<std::int32_t> cell_columns;
  for (std::size_t i = 0; i < cells.size(); ++i)
  {
    impl::cell_columns(
        std::span(columns).subspan(i * points_per_cell, points_per_cell),
        cell_columns);
    pattern.insert(dofmap1->cell_dofs(cells[i]), cell_columns);
  }
}

/// @brief Assemble the interpolation operator from a space on one mesh
/// to a space on another mesh.
///
/// If \f$u_0\f$ is the degree-of-freedom vector of a function in
/// \f$V_0\f$, then \f$u_1 = A u_0\f$ is the degree-of-freedom vector of
/// its interpolation into \f$V_1\f$ on `cells`.
///
/// The basis of \f$V_0\f$ is evaluated at points that are each given a
/// weight (see evaluate_basis_nonmatching). Interpolation point \f$p\f$
/// of a cell of \f$V_1\f$ corresponds to `nq` consecutive points, and
/// the values at these points, multiplied by their weights, are summed
/// before the interpolation operator of \f$V_1\f$ is applied. With one
/// point per interpolation point and unit weights this is the standard
/// interpolation. More points can express e.g. an average over a circle
/// around each interpolation point.
///
/// Rows are only added for degrees-of-freedom owned by this process.
/// Points that no process owns do not contribute.
///
/// @param[in] V0 Space to interpolate from.
/// @param[in] V1 Space to interpolate to.
/// @param[in] cells Cells of the mesh of `V1` to interpolate on. The
/// points are ordered by cell, then by interpolation point, then by the
/// `nq` points of each interpolation point.
/// @param[in] weights Weight of each point,
/// `shape=(cells.size() * num_interpolation_points * nq,)`.
/// @param[in] data Basis data from evaluate_basis_nonmatching.
/// @param[in] column_map Column index map from create_column_map.
/// @param[in] mat_add Functor that adds values to a matrix, with the
/// rows indexed by the dofmap of `V1` and the columns by `column_map`.
template <dolfinx::scalar T, std::floating_point U>
void interpolation_matrix(const FunctionSpace<U>& V0,
                          const FunctionSpace<U>& V1,
                          std::span<const std::int32_t> cells,
                          std::span<const U> weights,
                          const NonmatchingBasisData<U>& data,
                          const common::IndexMap& column_map, auto&& mat_add)
{
  std::shared_ptr<const FiniteElement<U>> e0 = V0.element();
  assert(e0);
  std::shared_ptr<const FiniteElement<U>> e1 = V1.element();
  assert(e1);
  if (!std::ranges::equal(e0->value_shape(), e1->value_shape()))
  {
    throw std::invalid_argument(
        "Interpolation operator: elements have different value dimensions");
  }

  std::shared_ptr<const DofMap> dofmap0 = V0.dofmap();
  assert(dofmap0);
  std::shared_ptr<const DofMap> dofmap1 = V1.dofmap();
  assert(dofmap1);
  const std::size_t num_cell_dofs0 = dofmap0->map().extent(1);
  const std::size_t num_points = data.dofs.size() / num_cell_dofs0;
  const auto [X, Xshape] = e1->interpolation_points();
  if (cells.empty() or Xshape[0] == 0)
    return;
  if (weights.size() != num_points
      or num_points % (cells.size() * Xshape[0]) != 0)
  {
    throw std::invalid_argument(
        "Number of weights and points is inconsistent with the cells.");
  }
  const std::size_t nq = num_points / (cells.size() * Xshape[0]);

  auto mesh = V1.mesh();
  assert(mesh);
  const std::size_t tdim = mesh->topology()->dim();
  const std::size_t gdim = mesh->geometry().dim();
  std::span<const std::uint32_t> cell_info;
  if (e1->needs_dof_transformations())
  {
    mesh->topology_mutable()->create_cell_permutations();
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }

  // Sizes. The basis of one block of V0 is unrolled for its block size,
  // as in the same-mesh interpolation_matrix.
  const int bs0 = e0->block_size();
  const int bs1 = e1->block_size();
  const std::size_t value_size0 = e0->physical_base_value_size();
  const std::size_t value_size_phys = e0->value_size();
  const std::size_t value_size_ref1
      = e1->reference_value_size() * static_cast<std::size_t>(bs1);
  const std::size_t space_dim1 = e1->space_dimension();

  // Coordinate element tabulated at the interpolation points of V1
  const CoordinateElement<U>& cmap = mesh->geometry().cmaps().front();
  auto x_dofmap = mesh->geometry().dofmaps().front();
  std::span<const U> x_g = mesh->geometry().x();
  std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, Xshape[0]);
  std::vector<U> phi_b(
      std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
  cmap.tabulate(1, X, Xshape, phi_b);
  md::mdspan<const U, md::dextents<std::size_t, 4>> phi(phi_b.data(),
                                                        phi_shape);

  const auto [_Pi_1, pi_shape] = e1->interpolation_operator();
  impl::mdspan_t<const U, 2> Pi_1(_Pi_1.data(), pi_shape);
  auto apply_Pi_1 = e1->interpolation_ident()
                        ? impl::interpolation_apply_columns<true, T, U>
                        : impl::interpolation_apply_columns<false, T, U>;
  auto apply_inverse_dof_transform1 = e1->template dof_transformation_fn<T>(
      doftransform::inverse_transpose, false);

  using u_t = md::mdspan<U, md::dextents<std::size_t, 2>>;
  using U_t = md::mdspan<const U, md::dextents<std::size_t, 2>>;
  using J_t = md::mdspan<const U, md::dextents<std::size_t, 2>>;
  using K_t = md::mdspan<const U, md::dextents<std::size_t, 2>>;
  auto pull_back_fn1
      = e1->basix_element().template map_fn<u_t, U_t, K_t, J_t>();

  // Working arrays
  std::vector<U> coord_dofs_b(cmap.dim() * gdim);
  md::mdspan<U, md::dextents<std::size_t, 2>> coord_dofs(coord_dofs_b.data(),
                                                         cmap.dim(), gdim);
  std::vector<U> J_b(Xshape[0] * gdim * tdim);
  md::mdspan<U, md::dextents<std::size_t, 3>> J(J_b.data(), Xshape[0], gdim,
                                                tdim);
  std::vector<U> K_b(Xshape[0] * tdim * gdim);
  md::mdspan<U, md::dextents<std::size_t, 3>> K(K_b.data(), Xshape[0], tdim,
                                                gdim);
  std::vector<U> detJ(Xshape[0]);
  std::vector<U> det_scratch(2 * gdim * tdim);
  std::vector<U> values_b, mapped_values_b;
  std::vector<T> Ab;

  const std::vector<std::int32_t> columns
      = impl::nonmatching_columns(data, column_map);
  std::vector<std::int32_t> cell_columns;
  const std::size_t points_per_cell = Xshape[0] * nq;

  // Only add each owned row once
  const int row_bs = dofmap1->bs();
  const std::int32_t num_owned_rows
      = dofmap1->index_map->size_local() * dofmap1->index_map_bs() / row_bs;
  const std::int32_t num_ghosted_rows
      = dofmap1->index_map->num_ghosts() * dofmap1->index_map_bs() / row_bs;
  std::vector<std::int8_t> row_added(num_owned_rows + num_ghosted_rows, 0);

  for (std::size_t ci = 0; ci < cells.size(); ++ci)
  {
    const std::int32_t c = cells[ci];
    auto point_columns
        = std::span(columns).subspan(ci * points_per_cell * num_cell_dofs0,
                                     points_per_cell * num_cell_dofs0);
    impl::cell_columns(point_columns, cell_columns);
    const std::size_t num_cols = cell_columns.size() * bs0;

    // Weighted sum of the basis values at the points of each
    // interpolation point, unrolled for the block size of V0
    values_b.assign(Xshape[0] * num_cols * value_size_phys, 0);
    impl::mdspan_t<U, 3> values(values_b.data(), Xshape[0], num_cols,
                                value_size_phys);
    for (std::size_t p = 0; p < Xshape[0]; ++p)
    {
      for (std::size_t k = 0; k < nq; ++k)
      {
        const std::size_t q = ci * points_per_cell + p * nq + k;
        const U w = weights[q];
        for (std::size_t i = 0; i < num_cell_dofs0; ++i)
        {
          std::int32_t col = columns[q * num_cell_dofs0 + i];
          if (col < 0)
            continue;
          const std::size_t m = std::ranges::distance(
              cell_columns.begin(),
              std::ranges::lower_bound(cell_columns, col));
          for (std::size_t j = 0; j < value_size0; ++j)
          {
            const U v
                = w * data.values[(q * num_cell_dofs0 + i) * value_size0 + j];
            for (int b = 0; b < bs0; ++b)
              values(p, m * bs0 + b, j * bs0 + b) += v;
          }
        }
      }
    }

    // Pull back to the reference cell of V1
    auto x_dofs = md::submdspan(x_dofmap, c, md::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
      for (std::size_t j = 0; j < gdim; ++j)
        coord_dofs(i, j) = x_g[3 * x_dofs[i] + j];
    impl::compute_jacobians<U>(cmap, phi, coord_dofs, J, K, detJ, det_scratch);
    mapped_values_b.assign(Xshape[0] * num_cols * value_size_ref1, 0);
    impl::mdspan_t<U, 3> mapped_values(mapped_values_b.data(), Xshape[0],
                                       num_cols, value_size_ref1);
    for (std::size_t p = 0; p < Xshape[0]; ++p)
    {
      auto _u = md::submdspan(values, p, md::full_extent, md::full_extent);
      auto _U
          = md::submdspan(mapped_values, p, md::full_extent, md::full_extent);
      auto _K = md::submdspan(K, p, md::full_extent, md::full_extent);
      auto _J = md::submdspan(J, p, md::full_extent, md::full_extent);
      pull_back_fn1(_U, _u, _K, 1.0 / detJ[p], _J);
    }

    // Apply the interpolation operator of V1 and transform for the
    // orientation of the cell
    Ab.resize(space_dim1 * num_cols);
    apply_Pi_1(Pi_1, mapped_values, bs1, Ab);
    if (apply_inverse_dof_transform1)
      apply_inverse_dof_transform1(Ab, cell_info, c, num_cols);

    // Zero rows that are ghosts or have already been added
    std::span<const std::int32_t> row_dofs = dofmap1->cell_dofs(c);
    for (std::size_t i = 0; i < row_dofs.size(); ++i)
    {
      std::int32_t r = row_dofs[i];
      if (r >= num_owned_rows or row_added[r])
      {
        for (int k = 0; k < row_bs; ++k)
        {
          std::fill_n(std::next(Ab.begin(), (i * row_bs + k) * num_cols),
                      num_cols, T(0));
        }
      }
      row_added[r] = 1;
    }

    mat_add(row_dofs, cell_columns, Ab);
  }
}

} // namespace dolfinx::fem
