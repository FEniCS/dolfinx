// Copyright (C) 2003-2022 Anders Logg, Garth N. Wells and Massimiliano Leoni
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "DofMap.h"
#include "FiniteElement.h"
#include "FunctionSpace.h"
#include "interpolate.h"
#include <concepts>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <functional>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace dolfinx::fem
{
template <dolfinx::scalar T, std::floating_point U>
class Expression;

/// This class represents a function \f$ u_h \f$ in a finite
/// element function space \f$ V_h \f$, given by
///
/// \f[ u_h = \sum_{i=1}^{n} U_i \phi_i, \f]
/// where \f$ \{\phi_i\}_{i=1}^{n} \f$ is a basis for \f$ V_h \f$,
/// and \f$ U \f$ is a vector of expansion coefficients for \f$ u_h \f$.
///
/// @tparam T The function scalar type.
/// @tparam U The mesh geometry scalar type.
template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
class Function
{
public:
  /// Field type for the Function, e.g. `double`, `std::complex<float>`,
  /// etc.
  using value_type = T;
  /// Geometry type of the Mesh that the Function is defined on.
  using geometry_type = U;

  /// Create function on given function space
  /// @param[in] V The function space
  explicit Function(std::shared_ptr<const FunctionSpace<geometry_type>> V)
      : _function_space(V),
        _x(std::make_shared<la::Vector<value_type>>(
            V->dofmap()->index_map, V->dofmap()->index_map_bs()))
  {
    if (!V->component().empty())
    {
      throw std::runtime_error("Cannot create Function from subspace. Consider "
                               "collapsing the function space");
    }
  }

  /// @brief Create function on given function space with a given
  /// vector.
  ///
  /// @warning This constructor is intended for internal library use
  /// only
  ///
  /// @param[in] V The function space
  /// @param[in] x The vector
  Function(std::shared_ptr<const FunctionSpace<geometry_type>> V,
           std::shared_ptr<la::Vector<value_type>> x)
      : _function_space(V), _x(x)
  {
    // We do not check for a subspace since this constructor is used for
    // creating subfunctions

    // Assertion uses '<=' to deal with sub-functions
    assert(V->dofmap());
    assert(V->dofmap()->index_map->size_global() * V->dofmap()->index_map_bs()
           <= _x->bs() * _x->index_map()->size_global());
  }

  // Copy constructor
  Function(const Function& v) = delete;

  /// Move constructor
  Function(Function&& v) = default;

  /// Destructor
  ~Function() = default;

  /// Move assignment
  Function& operator=(Function&& v) = default;

  // Assignment
  Function& operator=(const Function& v) = delete;

  /// @brief Extract a sub-function (a view into the Function).
  /// @param[in] i Index of subfunction
  /// @return The sub-function
  Function sub(int i) const
  {
    auto sub_space = std::make_shared<FunctionSpace<geometry_type>>(
        _function_space->sub({i}));
    assert(sub_space);
    return Function(sub_space, _x);
  }

  /// @brief Collapse a subfunction (view into a Function) to a
  /// stand-alone Function.
  /// @return New collapsed Function.
  Function collapse() const
  {
    // Create new collapsed FunctionSpace
    auto [V, map] = _function_space->collapse();

    // Create new vector
    auto x = std::make_shared<la::Vector<value_type>>(
        V.dofmap()->index_map, V.dofmap()->index_map_bs());

    // Copy values into new vector
    std::span<const value_type> x_old = _x->array();
    std::span<value_type> x_new = x->mutable_array();
    for (std::size_t i = 0; i < map.size(); ++i)
    {
      assert((int)i < x_new.size());
      assert(map[i] < x_old.size());
      x_new[i] = x_old[map[i]];
    }

    return Function(
        std::make_shared<FunctionSpace<geometry_type>>(std::move(V)), x);
  }

  /// @brief Access the function space.
  /// @return The function space
  std::shared_ptr<const FunctionSpace<geometry_type>> function_space() const
  {
    return _function_space;
  }

  /// @brief Underlying vector
  std::shared_ptr<const la::Vector<value_type>> x() const { return _x; }

  /// @brief Underlying vector
  std::shared_ptr<la::Vector<value_type>> x() { return _x; }

  /// @brief Interpolate a provided Function.
  /// @param[in] v The function to be interpolated
  /// @param[in] cells The cells to interpolate on
  /// @param[in] cell_map For cell `i` in the mesh associated with \p this,
  /// `cell_map[i]` is the index of the same cell, but in the mesh associated
  /// with `v`
  void interpolate(const Function<value_type, geometry_type>& v,
                   std::span<const std::int32_t> cells,
                   std::span<const std::int32_t> cell_map)
  {
    fem::interpolate(*this, v, cells, cell_map);
  }

  /// @brief Interpolate a provided Function.
  /// @param[in] v The function to be interpolated
  /// @param[in] cell_map Map from cells in self to cell indices in \p v
  void interpolate(const Function<value_type, geometry_type>& v,
                   std::span<const std::int32_t> cell_map = {})
  {
    assert(_function_space);
    assert(_function_space->mesh());
    int tdim = _function_space->mesh()->topology()->dim();
    auto cell_imap = _function_space->mesh()->topology()->index_map(tdim);
    assert(cell_imap);
    std::int32_t num_cells = cell_imap->size_local() + cell_imap->num_ghosts();
    std::vector<std::int32_t> cells(num_cells, 0);
    std::iota(cells.begin(), cells.end(), 0);
    interpolate(v, cells, cell_map);
  }

  /// Interpolate an expression function on a list of cells
  /// @param[in] f The expression function to be interpolated
  /// @param[in] cells The cells to interpolate on
  void interpolate(
      const std::function<
          std::pair<std::vector<value_type>, std::vector<std::size_t>>(
              MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
                  const geometry_type,
                  MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
                      std::size_t, 3,
                      MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent>>)>& f,
      std::span<const std::int32_t> cells)
  {
    assert(_function_space);
    assert(_function_space->element());
    assert(_function_space->mesh());
    const std::vector<geometry_type> x
        = fem::interpolation_coords<geometry_type>(
            *_function_space->element(), _function_space->mesh()->geometry(),
            cells);
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const geometry_type,
        MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
            std::size_t, 3, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent>>
        _x(x.data(), 3, x.size() / 3);

    const auto [fx, fshape] = f(_x);
    assert(fshape.size() <= 2);
    if (int vs = _function_space->value_size(); vs == 1 and fshape.size() == 1)
    {
      // Check for scalar-valued functions
      if (fshape.front() != x.size() / 3)
        throw std::runtime_error("Data returned by callable has wrong length");
    }
    else
    {
      // Check for vector/tensor value
      if (fshape.size() != 2)
        throw std::runtime_error("Expected 2D array of data");

      if (fshape[0] != vs)
      {
        throw std::runtime_error(
            "Data returned by callable has wrong shape(0) size");
      }

      if (fshape[1] != x.size() / 3)
      {
        throw std::runtime_error(
            "Data returned by callable has wrong shape(1) size");
      }
    }

    std::array<std::size_t, 2> _fshape;
    if (fshape.size() == 1)
      _fshape = {1, fshape[0]};
    else
      _fshape = {fshape[0], fshape[1]};

    fem::interpolate(*this, std::span<const value_type>(fx.data(), fx.size()),
                     _fshape, cells);
  }

  /// @brief Interpolate an expression function on the whole domain.
  /// @param[in] f Expression to be interpolated
  void
  interpolate(const std::function<
              std::pair<std::vector<value_type>, std::vector<std::size_t>>(
                  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
                      const geometry_type,
                      MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
                          std::size_t, 3,
                          MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent>>)>& f)
  {
    assert(_function_space);
    assert(_function_space->mesh());
    const int tdim = _function_space->mesh()->topology()->dim();
    auto cell_map = _function_space->mesh()->topology()->index_map(tdim);
    assert(cell_map);
    std::int32_t num_cells = cell_map->size_local() + cell_map->num_ghosts();
    std::vector<std::int32_t> cells(num_cells, 0);
    std::iota(cells.begin(), cells.end(), 0);
    interpolate(f, cells);
  }

  /// @brief Interpolate an Expression (based on UFL)
  /// @param[in] e Expression to be interpolated. The Expression
  /// must have been created using the reference coordinates
  /// `FiniteElement::interpolation_points()` for the element associated
  /// with `u`.
  /// @param[in] cells The cells to interpolate on
  /// @param[in] expr_mesh The mesh to evaluate the expression on
  /// @param[in] cell_map For cell `i` in the mesh associated with \p this,
  /// `cell_map[i]` is the index of the same cell, but in \p expr_mesh
  void interpolate(const Expression<value_type, geometry_type>& e,
                   std::span<const std::int32_t> cells,
                   const mesh::Mesh<geometry_type>& expr_mesh,
                   std::span<const std::int32_t> cell_map = {})
  {
    // Check that spaces are compatible
    assert(_function_space);
    assert(_function_space->element());
    std::size_t value_size = e.value_size();
    if (e.argument_function_space())
      throw std::runtime_error("Cannot interpolate Expression with Argument");

    if (value_size != _function_space->value_size())
    {
      throw std::runtime_error(
          "Function value size not equal to Expression value size");
    }

    {
      // Compatibility check
      auto [X0, shape0] = e.X();
      auto [X1, shape1] = _function_space->element()->interpolation_points();
      if (shape0 != shape1)
      {
        throw std::runtime_error(
            "Function element interpolation points has different shape to "
            "Expression interpolation points");
      }

      for (std::size_t i = 0; i < X0.size(); ++i)
      {
        if (std::abs(X0[i] - X1[i]) > 1.0e-10)
        {
          throw std::runtime_error("Function element interpolation points not "
                                   "equal to Expression interpolation points");
        }
      }
    }

    // Array to hold evaluated Expression
    std::size_t num_cells = cells.size();
    std::size_t num_points = e.X().second[0];
    std::vector<value_type> fdata(num_cells * num_points * value_size);
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const value_type,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>
        f(fdata.data(), num_cells, num_points, value_size);

    // Evaluate Expression at points
    assert(_function_space->mesh());

    std::vector<std::int32_t> cells_expr;
    cells_expr.reserve(num_cells);
    if (&expr_mesh == _function_space->mesh().get())
      cells_expr.insert(cells_expr.end(), cells.begin(), cells.end());
    else // If meshes are different and input mapping is given

    {
      std::transform(cells.begin(), cells.end(), std::back_inserter(cells_expr),
                     [&cell_map](std::int32_t c) { return cell_map[c]; });
    }

    e.eval(expr_mesh, cells_expr, fdata, {num_cells, num_points * value_size});

    // Reshape evaluated data to fit interpolate
    // Expression returns matrix of shape (num_cells, num_points *
    // value_size), i.e. xyzxyz ordering of dof values per cell per
    // point. The interpolation uses xxyyzz input, ordered for all
    // points of each cell, i.e. (value_size, num_cells*num_points)
    std::vector<value_type> fdata1(num_cells * num_points * value_size);
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        value_type, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>
        f1(fdata1.data(), value_size, num_cells, num_points);
    for (std::size_t i = 0; i < f.extent(0); ++i)
      for (std::size_t j = 0; j < f.extent(1); ++j)
        for (std::size_t k = 0; k < f.extent(2); ++k)
          f1(k, i, j) = f(i, j, k);

    // Interpolate values into appropriate space
    fem::interpolate(*this,
                     std::span<const value_type>(fdata1.data(), fdata1.size()),
                     {value_size, num_cells * num_points}, cells);
  }

  /// Interpolate an Expression (based on UFL) on all cells
  /// @param[in] e The function to be interpolated
  /// @param[in] expr_mesh Mesh the Expression `e` is defined on.
  /// @param[in] cell_map Map from `cells` to cells in expression if
  /// receiving function is defined on a different mesh than the expression
  void interpolate(const Expression<value_type, geometry_type>& e,
                   const mesh::Mesh<geometry_type>& expr_mesh,
                   std::span<const std::int32_t> cell_map
                   = std::span<const std::int32_t>() = {})
  {
    assert(_function_space);
    assert(_function_space->mesh());
    const int tdim = _function_space->mesh()->topology()->dim();
    auto cell_imap = _function_space->mesh()->topology()->index_map(tdim);
    assert(cell_imap);
    std::int32_t num_cells = cell_imap->size_local() + cell_imap->num_ghosts();
    std::vector<std::int32_t> cells(num_cells, 0);
    std::iota(cells.begin(), cells.end(), 0);
    interpolate(e, cells, expr_mesh, cell_map);
  }

  /// Interpolate a function defined on a non-matching mesh
  /// @param[in] v The function to be interpolated
  /// @param cells Cells in the mesh associated with `this` that will be
  /// interpolated into
  /// @param nmm_interpolation_data Data required for associating the
  /// interpolation points of `this` with cells in `v`. Can be computed with
  /// `fem::create_interpolation_data`.
  void
  interpolate(const Function<value_type, geometry_type>& v,
              std::span<const std::int32_t> cells,
              const geometry::PointOwnershipData<U>& nmm_interpolation_data)
  {
    fem::interpolate(*this, v, cells, nmm_interpolation_data);
  }

  /// @brief Evaluate the Function at points.
  /// @param[in] x The coordinates of the points. It has shape
  /// (num_points, 3) and storage is row-major.
  /// @param[in] xshape The shape of `x`.
  /// @param[in] cells An array of cell indices. cells[i] is the index
  /// of the cell that contains the point x(i). Negative cell indices
  /// can be passed, and the corresponding point will be ignored.
  /// @param[out] u The values at the points. Values are not computed
  /// for points with a negative cell index. This argument must be
  /// passed with the correct size. Storage is row-major.
  /// @param[in] ushape The shape of `u`.
  void eval(std::span<const geometry_type> x, std::array<std::size_t, 2> xshape,
            std::span<const std::int32_t> cells, std::span<value_type> u,
            std::array<std::size_t, 2> ushape) const
  {
    if (cells.empty())
      return;

    assert(x.size() == xshape[0] * xshape[1]);
    assert(u.size() == ushape[0] * ushape[1]);

    // TODO: This could be easily made more efficient by exploiting points
    // being ordered by the cell to which they belong.

    if (xshape[0] != cells.size())
    {
      throw std::runtime_error(
          "Number of points and number of cells must be equal.");
    }

    if (xshape[0] != ushape[0])
    {
      throw std::runtime_error(
          "Length of array for Function values must be the "
          "same as the number of points.");
    }

    // Get mesh
    assert(_function_space);
    auto mesh = _function_space->mesh();
    assert(mesh);
    const std::size_t gdim = mesh->geometry().dim();
    const std::size_t tdim = mesh->topology()->dim();
    auto map = mesh->topology()->index_map(tdim);

    // Get coordinate map
    const CoordinateElement<geometry_type>& cmap = mesh->geometry().cmap();

    // Get geometry data
    auto x_dofmap = mesh->geometry().dofmap();
    const std::size_t num_dofs_g = cmap.dim();
    auto x_g = mesh->geometry().x();

    // Get element
    auto element = _function_space->element();
    assert(element);
    const int bs_element = element->block_size();
    const std::size_t reference_value_size
        = element->reference_value_size() / bs_element;
    const std::size_t value_size = _function_space->value_size() / bs_element;
    const std::size_t space_dimension = element->space_dimension() / bs_element;

    // If the space has sub elements, concatenate the evaluations on the
    // sub elements
    const int num_sub_elements = element->num_sub_elements();
    if (num_sub_elements > 1 and num_sub_elements != bs_element)
    {
      throw std::runtime_error("Function::eval is not supported for mixed "
                               "elements. Extract subspaces.");
    }

    // Create work vector for expansion coefficients
    std::vector<value_type> coefficients(space_dimension * bs_element);

    // Get dofmap
    std::shared_ptr<const DofMap> dofmap = _function_space->dofmap();
    assert(dofmap);
    const int bs_dof = dofmap->bs();

    std::span<const std::uint32_t> cell_info;
    if (element->needs_dof_transformations())
    {
      mesh->topology_mutable()->create_entity_permutations();
      cell_info = std::span(mesh->topology()->get_cell_permutation_info());
    }

    std::vector<geometry_type> coord_dofs_b(num_dofs_g * gdim);
    impl::mdspan_t<geometry_type, 2> coord_dofs(coord_dofs_b.data(), num_dofs_g,
                                                gdim);
    std::vector<geometry_type> xp_b(1 * gdim);
    impl::mdspan_t<geometry_type, 2> xp(xp_b.data(), 1, gdim);

    // Loop over points
    std::fill(u.data(), u.data() + u.size(), 0.0);
    std::span<const value_type> _v = _x->array();

    // Evaluate geometry basis at point (0, 0, 0) on the reference cell.
    // Used in affine case.
    std::array<std::size_t, 4> phi0_shape = cmap.tabulate_shape(1, 1);
    std::vector<geometry_type> phi0_b(std::reduce(
        phi0_shape.begin(), phi0_shape.end(), 1, std::multiplies{}));
    impl::mdspan_t<const geometry_type, 4> phi0(phi0_b.data(), phi0_shape);
    cmap.tabulate(1, std::vector<geometry_type>(tdim), {1, tdim}, phi0_b);
    auto dphi0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        phi0, std::pair(1, tdim + 1), 0,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

    // Data structure for evaluating geometry basis at specific points.
    // Used in non-affine case.
    std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, 1);
    std::vector<geometry_type> phi_b(
        std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
    impl::mdspan_t<const geometry_type, 4> phi(phi_b.data(), phi_shape);
    auto dphi = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        phi, std::pair(1, tdim + 1), 0,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

    // Reference coordinates for each point
    std::vector<geometry_type> Xb(xshape[0] * tdim);
    impl::mdspan_t<geometry_type, 2> X(Xb.data(), xshape[0], tdim);

    // Geometry data at each point
    std::vector<geometry_type> J_b(xshape[0] * gdim * tdim);
    impl::mdspan_t<geometry_type, 3> J(J_b.data(), xshape[0], gdim, tdim);
    std::vector<geometry_type> K_b(xshape[0] * tdim * gdim);
    impl::mdspan_t<geometry_type, 3> K(K_b.data(), xshape[0], tdim, gdim);
    std::vector<geometry_type> detJ(xshape[0]);
    std::vector<geometry_type> det_scratch(2 * gdim * tdim);

    // Prepare geometry data in each cell
    for (std::size_t p = 0; p < cells.size(); ++p)
    {
      const int cell_index = cells[p];

      // Skip negative cell indices
      if (cell_index < 0)
        continue;

      // Get cell geometry (coordinate dofs)
      auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          x_dofmap, cell_index, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      assert(x_dofs.size() == num_dofs_g);
      for (std::size_t i = 0; i < num_dofs_g; ++i)
      {
        const int pos = 3 * x_dofs[i];
        for (std::size_t j = 0; j < gdim; ++j)
          coord_dofs(i, j) = x_g[pos + j];
      }

      for (std::size_t j = 0; j < gdim; ++j)
        xp(0, j) = x[p * xshape[1] + j];

      auto _J = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          J, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      auto _K = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          K, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

      std::array<geometry_type, 3> Xpb = {0, 0, 0};
      MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
          geometry_type,
          MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
              std::size_t, 1, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent>>
          Xp(Xpb.data(), 1, tdim);

      // Compute reference coordinates X, and J, detJ and K
      if (cmap.is_affine())
      {
        CoordinateElement<geometry_type>::compute_jacobian(dphi0, coord_dofs,
                                                           _J);
        CoordinateElement<geometry_type>::compute_jacobian_inverse(_J, _K);
        std::array<geometry_type, 3> x0 = {0, 0, 0};
        for (std::size_t i = 0; i < coord_dofs.extent(1); ++i)
          x0[i] += coord_dofs(0, i);
        CoordinateElement<geometry_type>::pull_back_affine(Xp, _K, x0, xp);
        detJ[p]
            = CoordinateElement<geometry_type>::compute_jacobian_determinant(
                _J, det_scratch);
      }
      else
      {
        // Pull-back physical point xp to reference coordinate Xp
        cmap.pull_back_nonaffine(Xp, xp, coord_dofs);

        cmap.tabulate(1, std::span(Xpb.data(), tdim), {1, tdim}, phi_b);
        CoordinateElement<geometry_type>::compute_jacobian(dphi, coord_dofs,
                                                           _J);
        CoordinateElement<geometry_type>::compute_jacobian_inverse(_J, _K);
        detJ[p]
            = CoordinateElement<geometry_type>::compute_jacobian_determinant(
                _J, det_scratch);
      }

      for (std::size_t j = 0; j < X.extent(1); ++j)
        X(p, j) = Xpb[j];
    }

    // Prepare basis function data structures
    std::vector<geometry_type> basis_derivatives_reference_values_b(
        1 * xshape[0] * space_dimension * reference_value_size);
    impl::mdspan_t<const geometry_type, 4> basis_derivatives_reference_values(
        basis_derivatives_reference_values_b.data(), 1, xshape[0],
        space_dimension, reference_value_size);
    std::vector<geometry_type> basis_values_b(space_dimension * value_size);
    impl::mdspan_t<geometry_type, 2> basis_values(basis_values_b.data(),
                                                  space_dimension, value_size);

    // Compute basis on reference element
    element->tabulate(basis_derivatives_reference_values_b, Xb,
                      {X.extent(0), X.extent(1)}, 0);

    using xu_t = impl::mdspan_t<geometry_type, 2>;
    using xU_t = impl::mdspan_t<const geometry_type, 2>;
    using xJ_t = impl::mdspan_t<const geometry_type, 2>;
    using xK_t = impl::mdspan_t<const geometry_type, 2>;
    auto push_forward_fn
        = element->basix_element().template map_fn<xu_t, xU_t, xJ_t, xK_t>();

    // Transformation function for basis function values
    auto apply_dof_transformation
        = element->template dof_transformation_fn<geometry_type>(
            doftransform::standard);

    // Size of tensor for symmetric elements, unused in non-symmetric case, but
    // placed outside the loop for pre-computation.
    int matrix_size;
    if (element->symmetric())
    {
      matrix_size = 0;
      while (matrix_size * matrix_size < ushape[1])
        ++matrix_size;
    }
    const std::size_t num_basis_values = space_dimension * reference_value_size;
    for (std::size_t p = 0; p < cells.size(); ++p)
    {
      const int cell_index = cells[p];
      if (cell_index < 0) // Skip negative cell indices
        continue;

      // Permute the reference basis function values to account for the
      // cell's orientation
      apply_dof_transformation(
          std::span(basis_derivatives_reference_values_b.data()
                        + p * num_basis_values,
                    num_basis_values),
          cell_info, cell_index, reference_value_size);

      {
        auto _U = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            basis_derivatives_reference_values, 0, p,
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        auto _J = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            J, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        auto _K = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            K, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        push_forward_fn(basis_values, _U, _J, detJ[p], _K);
      }

      // Get degrees of freedom for current cell
      std::span<const std::int32_t> dofs = dofmap->cell_dofs(cell_index);
      for (std::size_t i = 0; i < dofs.size(); ++i)
        for (int k = 0; k < bs_dof; ++k)
          coefficients[bs_dof * i + k] = _v[bs_dof * dofs[i] + k];

      if (element->symmetric())
      {
        int row = 0;
        int rowstart = 0;
        // Compute expansion
        for (int k = 0; k < bs_element; ++k)
        {
          if (k - rowstart > row)
          {
            row++;
            rowstart = k;
          }
          for (std::size_t i = 0; i < space_dimension; ++i)
          {
            for (std::size_t j = 0; j < value_size; ++j)
            {
              u[p * ushape[1]
                + (j * bs_element + row * matrix_size + k - rowstart)]
                  += coefficients[bs_element * i + k] * basis_values(i, j);
              if (k - rowstart != row)
              {
                u[p * ushape[1]
                  + (j * bs_element + row + matrix_size * (k - rowstart))]
                    += coefficients[bs_element * i + k] * basis_values(i, j);
              }
            }
          }
        }
      }
      else
      {
        // Compute expansion
        for (int k = 0; k < bs_element; ++k)
        {
          for (std::size_t i = 0; i < space_dimension; ++i)
          {
            for (std::size_t j = 0; j < value_size; ++j)
            {
              u[p * ushape[1] + (j * bs_element + k)]
                  += coefficients[bs_element * i + k] * basis_values(i, j);
            }
          }
        }
      }
    }
  }

  /// Name
  std::string name = "u";

private:
  // The function space
  std::shared_ptr<const FunctionSpace<geometry_type>> _function_space;

  // The vector of expansion coefficients (local)
  std::shared_ptr<la::Vector<value_type>> _x;
};

} // namespace dolfinx::fem
