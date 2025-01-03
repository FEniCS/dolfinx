// Copyright (C) 2015-2022 Garth N. Wells, Jørgen S. Dokken
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
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <span>
#include <vector>

namespace dolfinx::fem
{

/// @brief Assemble a discrete curl operator.
///
/// Curl of Nedelec(k) function -> RT(k-1) function.
///
/// @param[in] topology Mesh topology
/// @param[in] V0 Lagrange element and dofmap for corresponding space to
/// interpolate the gradient from
/// @param[in] V1 Nédélec (first kind) element and and dofmap for
/// corresponding space to interpolate into
/// @param[in] mat_set A functor that sets values in a matrix
template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
void discrete_curl(const FunctionSpace<U>& V0, const FunctionSpace<U>& V1,
                   auto&& mat_set)
{
  namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;
  using mdspan2_t = md::mdspan<U, md::dextents<std::size_t, 2>>;
  using mdspan3_t = md::mdspan<U, md::dextents<std::size_t, 3>>;
  using mdspan4_t = md::mdspan<U, md::dextents<std::size_t, 4>>;

  using cmdspan2_t = md::mdspan<const U, md::dextents<std::size_t, 2>>;
  using cmdspan4_t = md::mdspan<const U, md::dextents<std::size_t, 4>>;

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

  // Get dof transformation operators
  auto apply_dof_transformation0
      = e0->template dof_transformation_fn<U>(doftransform::standard, false);
  auto apply_inverse_dof_transform1 = e1->template dof_transformation_fn<T>(
      doftransform::inverse_transpose, false);

  // // Get sizes of elements
  const std::size_t space_dim0 = e0->space_dimension();
  const std::size_t space_dim1 = e1->space_dimension();
  const std::size_t value_size0 = e0->reference_value_size();
  const std::size_t value_size1 = e1->reference_value_size();

  // Evaluate coordinate map basis at reference interpolation points
  const auto [X, Xshape] = e1->interpolation_points();
  // const std::size_t num_points = Xshape[0];

  // Get/compute geometry map data
  const CoordinateElement<U>& cmap = mesh->geometry().cmap();
  auto x_dofmap = mesh->geometry().dofmap();
  const std::size_t num_dofs_g = cmap.dim();
  std::span<const U> x_g = mesh->geometry().x();
  std::array<std::size_t, 4> Phi_g_shape = cmap.tabulate_shape(1, Xshape[0]);
  std::vector<U> Phi_g_b(std::reduce(Phi_g_shape.begin(), Phi_g_shape.end(), 1,
                                     std::multiplies{}));
  cmdspan4_t Phi_g(Phi_g_b.data(), Phi_g_shape);
  cmap.tabulate(1, X, Xshape, Phi_g_b);

  // Geometry data structures
  std::vector<U> coord_dofs_b(num_dofs_g * gdim);
  mdspan2_t coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);
  std::vector<U> J_b(Xshape[0] * gdim * tdim);
  mdspan3_t J(J_b.data(), Xshape[0], gdim, tdim);
  std::vector<U> K_b(Xshape[0] * tdim * gdim);
  mdspan3_t K(K_b.data(), Xshape[0], tdim, gdim);
  std::vector<U> detJ(Xshape[0]);
  std::vector<U> det_scratch(2 * tdim * gdim);

  // Evaluate V0 basis function derivatives at reference interpolation
  // points for V1
  auto [Phi0_b, Phi0_shape] = e0->tabulate(X, Xshape, 1);
  cmdspan4_t Phi0(Phi0_b.data(), Phi0_shape);
  e0->tabulate(Phi0_b, X, Xshape, 1); // (deriv, pt_idx, phi_idx, comp)

  // Clamp values
  std::ranges::transform(Phi0_b, Phi0_b.begin(), [atol = 1e-14](auto x)
                         { return std::abs(x) < atol ? 0 : x; });

  // Create working arrays Phi0: (deriv, pt_idx, phi_idx/dof, comp)
  //                      _dPhi: (point, phi, deriv, comp)
  std::vector<U> _dPhi0_b(Phi0.extent(1) * Phi0.extent(2) * (Phi0.extent(1) - 1)
                          * Phi0.extent(3));
  mdspan4_t _dPhi0(_dPhi0_b.data(), Phi0.extent(1), Phi0.extent(2),
                   (Phi0.extent(1) - 1),
                   Phi0.extent(3)); // (point, phi, deriv, comp)

  std::vector<U> tmp_Phi0_b(Phi0.extent(1) * Phi0.extent(2) * Phi0.extent(3));
  mdspan3_t tmp_Phi0(tmp_Phi0_b.data(), Phi0.extent(1), Phi0.extent(2),
                     Phi0.extent(3)); // (point, phi, comp)

  // Create working arrays Phi0: (pt_idx, phi_idx/dof, deriv, comp)
  std::vector<U> dphi0_int_b(_dPhi0.extent(0) * _dPhi0.extent(1)
                             * _dPhi0.extent(2) * _dPhi0.extent(3));
  mdspan4_t dphi0_int(dphi0_int_b.data(), _dPhi0.extents());

  // Create working arrays Phi0: (pt_idx, phi_idx/dof, deriv, comp)
  std::vector<U> tmp_phi0_int_b(tmp_Phi0.extent(0) * tmp_Phi0.extent(1)
                                * tmp_Phi0.extent(2));
  mdspan3_t tmp_phi0_int(tmp_phi0_int_b.data(), tmp_Phi0.extents());

  std::vector<U> dphi0_b(_dPhi0.extent(0) * _dPhi0.extent(1) * _dPhi0.extent(2)
                         * _dPhi0.extent(3));
  mdspan4_t dphi0(dphi0_b.data(), _dPhi0.extents());

  std::vector<U> tmp_phi0_b(tmp_Phi0.extent(0) * tmp_Phi0.extent(1)
                            * tmp_Phi0.extent(2));
  mdspan3_t tmp_phi0(tmp_phi0_b.data(), tmp_Phi0.extents());

  // Get the interpolation operator (matrix) `Pi` that maps a function
  // evaluated at the interpolation points to the V1 element degrees of
  // freedom, i.e. dofs = Pi f_x
  const auto [Pi1_b, pi_shape] = e1->interpolation_operator();
  cmdspan2_t Pi_1(Pi1_b.data(), pi_shape);

  // curl data structures (pt_idx, phi_idx/dof, comp)
  std::vector<U> curl0_b(_dPhi0.extent(0) * _dPhi0.extent(1)
                         * _dPhi0.extent(3));
  mdspan3_t curl0(curl0_b.data(), _dPhi0.extent(0), _dPhi0.extent(1),
                  _dPhi0.extent(3));
  std::vector<U> Curl1_b(_dPhi0.extent(0) * _dPhi0.extent(1)
                         * _dPhi0.extent(3));
  mdspan3_t Curl1(Curl1_b.data(), _dPhi0.extent(0), _dPhi0.extent(1),
                  _dPhi0.extent(3));

  std::vector<T> Ab(space_dim0 * space_dim1);
  std::vector<T> local1(space_dim1);

  using u_t = md::mdspan<U, md::dextents<std::size_t, 2>>;
  using U_t = md::mdspan<const U, md::dextents<std::size_t, 2>>;
  using J_t = md::mdspan<const U, md::dextents<std::size_t, 2>>;
  using K_t = md::mdspan<const U, md::dextents<std::size_t, 2>>;
  auto push_forward_fn0
      = e0->basix_element().template map_fn<u_t, U_t, J_t, K_t>();

  // Iterate over mesh and interpolate on each cell
  auto cell_map = mesh->topology()->index_map(tdim);
  assert(cell_map);
  std::int32_t num_cells = cell_map->size_local();
  for (std::int32_t c = 0; c < num_cells; ++c)
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
          = md::submdspan(Phi_g, std::pair(1, tdim + 1), p, md::full_extent, 0);
      auto _J = md::submdspan(J, p, md::full_extent, md::full_extent);
      cmap.compute_jacobian(dPhi_g, coord_dofs, _J);
      auto _K = md::submdspan(K, p, md::full_extent, md::full_extent);
      cmap.compute_jacobian_inverse(_J, _K);
      detJ[p] = cmap.compute_jacobian_determinant(_J, det_scratch);
    }

    // TODO: re-order loops and/or re-pack Phi0

    // Copy (d)Phi0 (on reference) and apply DOF transformation
    // Phi0:   (deriv, pt_idx, phi_idx, comp)
    // _dPhi0: (pt_idx, phi_idx, deriv, comp)
    for (std::size_t p = 0; p < Phi0.extent(1); ++p)            // point
      for (std::size_t phi = 0; phi < Phi0.extent(2); ++phi)    // phi_i
        for (std::size_t d = 0; d < Phi0.extent(3); ++d)        // Comp. of phi
          for (std::size_t dx = 0; dx < _dPhi0.extent(2); ++dx) // dx
          {
            _dPhi0(p, phi, dx, d) = Phi0(dx + 1, p, phi, d);
            // std::cout << "Test: " << _dPhi0(p, phi, dx, d) << std::endl;
          }

    for (std::size_t p = 0; p < Phi0.extent(1); ++p)         // point
      for (std::size_t phi = 0; phi < Phi0.extent(2); ++phi) // phi_i
        for (std::size_t d = 0; d < Phi0.extent(3); ++d)     // Comp. of phi
          tmp_Phi0(p, phi, d) = Phi0(0, p, phi, d);

    for (std::size_t p = 0; p < _dPhi0.extent(0); ++p) // point
    {
      // Size: num_phi * num_derivs * num_components
      std::size_t size = _dPhi0.extent(1) * _dPhi0.extent(2) * _dPhi0.extent(3);
      std::size_t offset = p * size; // Offset for point p

      // Shape: (num_phi , (value_size * num_derivs))
      apply_dof_transformation0(std::span(_dPhi0.data_handle() + offset, size),
                                cell_info, c,
                                _dPhi0.extent(2) * _dPhi0.extent(3));
    }

    for (std::size_t p = 0; p < Xshape[0]; ++p)
    {
      apply_dof_transformation0(
          std::span(tmp_Phi0.data_handle() + p * space_dim0 * 3,
                    space_dim0 * 3),
          cell_info, c, 3);
    }

    // Copy evaluated basis on reference and push forward to physical
    // element
    for (std::size_t p = 0; p < _dPhi0.extent(0); ++p) // points
    {
      auto _J = md::submdspan(J, p, md::full_extent, md::full_extent);
      auto _K = md::submdspan(K, p, md::full_extent, md::full_extent);

      // _dPhi0: (pt_idx, phi_idx, deriv, comp)
      // XXX Push forward (contravariant Piola), (1/|J|) * J * phi
      // Push forward (covariant Piola),  J^{-T} * phi
      // Note, push_forward_fn0(_u, _U, _J, detJ[p], _K) can't handle
      // strides (yet)
      for (std::size_t b = 0; b < dphi0_int.extent(1); ++b) // basis index
      {
        for (std::size_t d = 0; d < dphi0_int.extent(2); ++d) // derivative
        {
          // std::pair<std::size_t, std::size_t> block{3 * d, 3 * d + 3};
          // auto _u0 = md::submdspan(dphi0_int, p, b, d, block);
          // auto _U = md::submdspan(_dPhi0, p, b, d, block);
          auto _u0 = md::submdspan(dphi0_int, p, b, d, md::full_extent);
          auto _U = md::submdspan(_dPhi0, p, b, d, md::full_extent);
          for (std::size_t i = 0; i < _J.extent(0); ++i)
          {
            _u0(i) = 0;
            for (std::size_t j = 0; j < _J.extent(1); ++j)
              _u0(i) += _K(j, i) * _U(j);
            // _u0(i) += _J(i, j) * _U(j);
            // _u0(i) /= detJ[p];
          }
        }
      }

      // Testing
      {
        // (point, phi, comp)
        std::vector<U> xtmp_Phi0_b(tmp_Phi0_b);
        mdspan3_t xtmp_Phi0(xtmp_Phi0_b.data(), tmp_Phi0.extents());

        std::vector<U> xtmp_phi0_int_b(tmp_Phi0_b);
        mdspan3_t xtmp_phi0_int(xtmp_phi0_int_b.data(), tmp_Phi0.extents());

        for (std::size_t b = 0; b < tmp_phi0_int.extent(1); ++b) // basis index
        {
          auto _U = md::submdspan(tmp_Phi0, p, b, md::full_extent);
          auto _u0 = md::submdspan(tmp_phi0_int, p, b, md::full_extent);
          for (std::size_t i = 0; i < _J.extent(0); ++i)
          {
            // _u0(i) = U(i);
            _u0(i) = 0;
            for (std::size_t j = 0; j < _J.extent(1); ++j)
              _u0(i) += _K(j, i) * _U(j);
          }
        }

        auto _U = md::submdspan(xtmp_Phi0, p, md::full_extent, md::full_extent);
        auto _u
            = md::submdspan(xtmp_phi0_int, p, md::full_extent, md::full_extent);
        auto _K = md::submdspan(K, p, md::full_extent, md::full_extent);
        auto _J = md::submdspan(J, p, md::full_extent, md::full_extent);
        push_forward_fn0(_u, _U, _J, detJ[p], _K);

        // std::cout << "Testing push-forward" << std::endl;
        // for (std::size_t i = 0; i < _u.extent(0); ++i)
        // {
        //   for (std::size_t j = 0; j < _u.extent(1); ++j)
        //   {
        //     std::cout << _U(i, j) << ", " << _u(i, j) << ", "
        //               << tmp_phi0_int(p, i, j) << std::endl;
        //   }
        // }
      }

      // Push forward derivatives
      for (std::size_t b = 0; b < dphi0_int.extent(1); ++b) // basis index
      {
        // std::pair<std::size_t, std::size_t> block{
        //     3 * dphi0_int.extent(2), 3 * (dphi0_int.extent(2) + 1)};
        auto du0 = md::submdspan(dphi0_int, p, b, md::full_extent,
                                 md::full_extent); // (deriv, comp)
        auto du = md::submdspan(dphi0, p, b, md::full_extent, md::full_extent);
        for (std::size_t i = 0; i < _K.extent(0); ++i) // component of phi
        {
          for (std::size_t j = 0; j < _K.extent(1); ++j) // derivative
          {
            du(j, i) = 0;
            for (std::size_t k = 0; k < _K.extent(0); ++k)
            {
              // dphi_i/dx_j = (dPhi_i/dX_k) (dX_k/ dx_j)
              du(j, i) += du0(k, i) * _K(k, j);
            }
          }
        }
      }
    }

    // Compute curl on physical element
    // dphi0: (pt_idx, phi_idx, deriv, comp)
    // curl: (pt_idx, phi_idx, comp)
    // for (std::size_t p = 0; p < curl0.extent(0); ++p) // point
    for (std::size_t p = 0; p < tmp_phi0_int.extent(0); ++p) // point
    {
      for (std::size_t i = 0; i < curl0.extent(1); ++i) // phi_i
      {
        // Note: should be dphi0
        // std::cout << "dphi_int: " << dphi0_int(p, i, 1, 2) << std::endl;
        curl0(p, i, 0) = dphi0(p, i, 1, 2) - dphi0(p, i, 2, 1);
        curl0(p, i, 1) = dphi0(p, i, 2, 0) - dphi0(p, i, 0, 2);
        curl0(p, i, 2) = dphi0(p, i, 0, 1) - dphi0(p, i, 1, 0);
      }

      if (c == 0)
      {
        // Constant (1^3)
        // std::vector<T> dofs{
        //     1.00000000e+00,  0.00000000e+00, 1.00000000e+00,  0.00000000e+00,
        //     -2.03998319e-17, 0.00000000e+00, 1.00000000e+00,  0.00000000e+00,
        //     -6.40268308e-18, 0.00000000e+00, -1.83417455e-17, 0.00000000e+00,
        //     -1.44248595e-17, 7.07106781e-01, -4.52738063e-18, 7.07106781e-01,
        //     -1.29695726e-17, 7.07106781e-01, -4.52738063e-18,
        //     -1.89522401e-17};

        // Linear (1^3)
        // std::vector<T> dofs{
        //     1.00000000e+00, 0.00000000e+00,  5.00000000e-01,  2.88675135e-01,
        //     -1.01999160e-17,
        //     -5.88892423e-18, 5.00000000e-01,  2.88675135e-01,
        //     -3.20134154e-18, -1.84829540e-18, 1.01817141e-33, 0.00000000e+00,
        //     -9.61657300e-18, 4.71404521e-01, -3.01825375e-18, 4.71404521e-01,
        //     -4.32319088e-18, 2.35702260e-01,  -1.50912688e-18,
        //     -6.31741338e-18};

        // Linear (2^3)
        std::vector<T> dofs{
            2.50000000e-01,  1.38777878e-17,  1.25000000e-01,  7.21687836e-02,
            -2.54997899e-18, -1.47223106e-18, 1.25000000e-01,  7.21687836e-02,
            -8.00335385e-19, -4.62073850e-19, 2.54542853e-34,  0.00000000e+00,
            -2.40414325e-18, 1.17851130e-01,  -7.54563438e-19, 1.17851130e-01,
            -1.08079772e-18, 5.89255651e-02,  -3.77281719e-19, -1.57935334e-18};

        std::vector<T> val(3, 0);
        // for (std::size_t i = 0; i < curl0.extent(1); ++i) // phi_i
        for (std::size_t i = 0; i < tmp_phi0_int.extent(1); ++i) // phi_i
        {
          val[0] += tmp_phi0_int(p, i, 0) * dofs[i];
          val[1] += tmp_phi0_int(p, i, 1) * dofs[i];
          val[2] += tmp_phi0_int(p, i, 2) * dofs[i];
        }
        std::cout << std::endl << "Value at point: " << p << std::endl;
        std::cout << val[0] << ", " << val[1] << ", " << val[2] << std::endl;

        std::vector<T> curl_val(3, 0);
        for (std::size_t i = 0; i < curl0.extent(1); ++i) // phi_i
        {
          curl_val[0] += curl0(p, i, 0) * dofs[i];
          curl_val[1] += curl0(p, i, 1) * dofs[i];
          curl_val[2] += curl0(p, i, 2) * dofs[i];
        }
        std::cout << std::endl << "Curl at point: " << p << std::endl;
        std::cout << curl_val[0] << ", " << curl_val[1] << ", " << curl_val[2]
                  << std::endl;
      }
    }

    // XXX Pull-back curl to V1 reference (inverse covariant map, J^T)
    // Pull-back curl to V1 reference (inverse contravariant map, det(J) J^{-1})
    // dphi0: (pt_idx, phi_idx, deriv, comp)
    // curl: (pt_idx, phi_idx, comp)
    for (std::size_t p = 0; p < curl0.extent(0); ++p) // point
    {
      for (std::size_t b = 0; b < curl0.extent(1); ++b) // phi_i
      {
        for (std::size_t i = 0; i < curl0.extent(2); ++i) // component
        {
          Curl1(p, b, i) = 0;
          for (std::size_t j = 0; j < curl0.extent(2); ++j)
            Curl1(p, b, i) += detJ[p] * K(p, i, j) * curl0(p, b, j);
          // Curl1(p, b, i) += J(p, j, i) * curl0(p, b, j);
        }
      }
    }

    // Apply interpolation matrix to basis values of V0 at the
    // interpolation points of V1
    // md::mdspan<T, md::dextents<std::size_t, 2>> A(Ab.data(), space_dim1,
    //                                               space_dim0);
    // for (std::size_t comp = 0; comp < 3; ++comp)
    // {
    //   for (std::size_t i = 0; i < space_dim1; ++i)
    //   {
    //     for (std::size_t j = 0; j < Xshape[0]; ++j)
    //     {
    //       A(i, j + comp * Xshape[0]) = 0;
    //       for (std::size_t k = 0; k < Xshape[0]; ++k)
    //       {
    //         // std::cout << "Test: " << Pi_1(i, k + comp * Xshape[0]) << ", "
    //         //           << Curl1(k, i, comp) << std::endl;
    //         A(i, j + comp * Xshape[0])
    //             += Pi_1(i, k + comp * Xshape[0]) * Curl1(k, i, comp);
    //       }
    //     }
    //   }
    // }

    for (std::size_t i = 0; i < Curl1.extent(1); ++i)
    {
      auto values = md::submdspan(Curl1, md::full_extent, i, md::full_extent);
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
/// @param[in] topology Mesh topology
/// @param[in] V0 Lagrange element and dofmap for corresponding space to
/// interpolate the gradient from
/// @param[in] V1 Nédélec (first kind) element and and dofmap for
/// corresponding space to interpolate into
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
/// @param[in] V0 The space to interpolate from
/// @param[in] V1 The space to interpolate to
/// @param[in] mat_set A functor that sets values in a matrix
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
