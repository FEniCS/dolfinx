// Copyright (C) 2003-2022 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "DofMap.h"
#include "FiniteElement.h"
#include "FunctionSpace.h"
#include "interpolate.h"
#include <dolfinx/common/IndexMap.h>
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
class FunctionSpace;
template <typename T>
class Expression;

template <typename T>
class Expression;

/// This class represents a function \f$ u_h \f$ in a finite
/// element function space \f$ V_h \f$, given by
///
/// \f[     u_h = \sum_{i=1}^{n} U_i \phi_i \f]
/// where \f$ \{\phi_i\}_{i=1}^{n} \f$ is a basis for \f$ V_h \f$,
/// and \f$ U \f$ is a vector of expansion coefficients for \f$ u_h \f$.
template <typename T>
class Function
{
public:
  /// Field type for the Function, e.g. double
  using value_type = T;

  /// Create function on given function space
  /// @param[in] V The function space
  explicit Function(std::shared_ptr<const FunctionSpace> V)
      : _function_space(V),
        _x(std::make_shared<la::Vector<T>>(V->dofmap()->index_map,
                                           V->dofmap()->index_map_bs()))
  {
    if (!V->component().empty())
    {
      throw std::runtime_error("Cannot create Function from subspace. Consider "
                               "collapsing the function space");
    }
  }

  /// Create function on given function space with a given vector
  ///
  /// @warning This constructor is intended for internal library use
  /// only
  ///
  /// @param[in] V The function space
  /// @param[in] x The vector
  Function(std::shared_ptr<const FunctionSpace> V,
           std::shared_ptr<la::Vector<T>> x)
      : _function_space(V), _x(x)
  {
    // We do not check for a subspace since this constructor is used for
    // creating subfunctions

    // Assertion uses '<=' to deal with sub-functions
    assert(V->dofmap());
    assert(V->dofmap()->index_map->size_global() * V->dofmap()->index_map_bs()
           <= _x->bs() * _x->map()->size_global());
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

  /// Extract subfunction (view into the Function)
  /// @param[in] i Index of subfunction
  /// @return The subfunction
  Function sub(int i) const
  {
    auto sub_space = _function_space->sub({i});
    assert(sub_space);
    return Function(sub_space, _x);
  }

  /// Collapse a subfunction (view into a Function) to a stand-alone
  /// Function
  /// @return New collapsed Function
  Function collapse() const
  {
    // Create new collapsed FunctionSpace
    auto [V, map] = _function_space->collapse();

    // Create new vector
    auto x = std::make_shared<la::Vector<T>>(V.dofmap()->index_map,
                                             V.dofmap()->index_map_bs());

    // Copy values into new vector
    std::span<const T> x_old = _x->array();
    std::span<T> x_new = x->mutable_array();
    for (std::size_t i = 0; i < map.size(); ++i)
    {
      assert((int)i < x_new.size());
      assert(map[i] < x_old.size());
      x_new[i] = x_old[map[i]];
    }

    return Function(std::make_shared<FunctionSpace>(std::move(V)), x);
  }

  /// Access the function space
  /// @return The function space
  std::shared_ptr<const FunctionSpace> function_space() const
  {
    return _function_space;
  }

  /// Underlying vector
  std::shared_ptr<const la::Vector<T>> x() const { return _x; }

  /// Underlying vector
  std::shared_ptr<la::Vector<T>> x() { return _x; }

  /// Interpolate a Function
  /// @param[in] v The function to be interpolated
  /// @param[in] cells The cells to interpolate on
  void interpolate(const Function<T>& v, std::span<const std::int32_t> cells)
  {
    fem::interpolate(*this, v, cells);
  }

  /// Interpolate a Function
  /// @param[in] v The function to be interpolated
  void interpolate(const Function<T>& v)
  {
    assert(_function_space);
    assert(_function_space->mesh());
    int tdim = _function_space->mesh()->topology().dim();
    auto cell_map = _function_space->mesh()->topology().index_map(tdim);
    assert(cell_map);
    std::int32_t num_cells = cell_map->size_local() + cell_map->num_ghosts();
    std::vector<std::int32_t> cells(num_cells, 0);
    std::iota(cells.begin(), cells.end(), 0);

    fem::interpolate(*this, v, cells);
  }

  /// Interpolate an expression function on a list of cells
  /// @param[in] f The expression function to be interpolated
  /// @param[in] cells The cells to interpolate on
  void interpolate(
      const std::function<std::pair<std::vector<T>, std::vector<std::size_t>>(
          std::experimental::mdspan<
              const double,
              std::experimental::extents<
                  std::size_t, 3, std::experimental::dynamic_extent>>)>& f,
      std::span<const std::int32_t> cells)
  {
    assert(_function_space);
    assert(_function_space->element());
    assert(_function_space->mesh());
    const std::vector<double> x = fem::interpolation_coords(
        *_function_space->element(), *_function_space->mesh(), cells);
    namespace stdex = std::experimental;
    stdex::mdspan<const double,
                  stdex::extents<std::size_t, 3, stdex::dynamic_extent>>
        _x(x.data(), 3, x.size() / 3);

    const auto [fx, fshape] = f(_x);
    assert(fshape.size() <= 2);
    if (int vs = _function_space->element()->value_size();
        vs == 1 and fshape.size() == 1)
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

    fem::interpolate(*this, std::span<const T>(fx.data(), fx.size()), _fshape,
                     cells);
  }

  /// Interpolate an expression function on the whole domain
  /// @param[in] f The expression to be interpolated
  void interpolate(
      const std::function<std::pair<std::vector<T>, std::vector<std::size_t>>(
          std::experimental::mdspan<
              const double,
              std::experimental::extents<
                  std::size_t, 3, std::experimental::dynamic_extent>>)>& f)
  {
    assert(_function_space);
    assert(_function_space->mesh());
    const int tdim = _function_space->mesh()->topology().dim();
    auto cell_map = _function_space->mesh()->topology().index_map(tdim);
    assert(cell_map);
    std::int32_t num_cells = cell_map->size_local() + cell_map->num_ghosts();
    std::vector<std::int32_t> cells(num_cells, 0);
    std::iota(cells.begin(), cells.end(), 0);
    interpolate(f, cells);
  }

  /// Interpolate an Expression (based on UFL)
  /// @param[in] e The Expression to be interpolated. The Expression
  /// must have been created using the reference coordinates
  /// `FiniteElement::interpolation_points()` for the element associated
  /// with `u`.
  /// @param[in] cells The cells to interpolate on
  void interpolate(const Expression<T>& e, std::span<const std::int32_t> cells)
  {
    // Check that spaces are compatible
    assert(_function_space);
    assert(_function_space->element());
    std::size_t value_size = e.value_size();
    if (e.argument_function_space())
      throw std::runtime_error("Cannot interpolate Expression with Argument");

    if (value_size != _function_space->element()->value_size())
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

    namespace stdex = std::experimental;

    // Array to hold evaluated Expression
    std::size_t num_cells = cells.size();
    std::size_t num_points = e.X().second[0];
    std::vector<T> fdata(num_cells * num_points * value_size);
    stdex::mdspan<const T, stdex::dextents<std::size_t, 3>> f(
        fdata.data(), num_cells, num_points, value_size);

    // Evaluate Expression at points
    e.eval(cells, fdata, {num_cells, num_points * value_size});

    // Reshape evaluated data to fit interpolate
    // Expression returns matrix of shape (num_cells, num_points *
    // value_size), i.e. xyzxyz ordering of dof values per cell per point.
    // The interpolation uses xxyyzz input, ordered for all points of each
    // cell, i.e. (value_size, num_cells*num_points)

    std::vector<T> fdata1(num_cells * num_points * value_size);
    stdex::mdspan<T, stdex::dextents<std::size_t, 3>> f1(
        fdata1.data(), value_size, num_cells, num_points);
    for (std::size_t i = 0; i < f.extent(0); ++i)
      for (std::size_t j = 0; j < f.extent(1); ++j)
        for (std::size_t k = 0; k < f.extent(2); ++k)
          f1(k, i, j) = f(i, j, k);

    // Interpolate values into appropriate space
    fem::interpolate(*this, std::span<const T>(fdata1.data(), fdata1.size()),
                     {value_size, num_cells * num_points}, cells);
  }

  /// Interpolate an Expression (based on UFL) on all cells
  /// @param[in] e The function to be interpolated
  void interpolate(const Expression<T>& e)
  {
    assert(_function_space);
    assert(_function_space->mesh());
    const int tdim = _function_space->mesh()->topology().dim();
    auto cell_map = _function_space->mesh()->topology().index_map(tdim);
    assert(cell_map);
    std::int32_t num_cells = cell_map->size_local() + cell_map->num_ghosts();
    std::vector<std::int32_t> cells(num_cells, 0);
    std::iota(cells.begin(), cells.end(), 0);
    interpolate(e, cells);
  }

  /// @brief Evaluate the Function at points.
  ///
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
  void eval(std::span<const double> x, std::array<std::size_t, 2> xshape,
            std::span<const std::int32_t> cells, std::span<T> u,
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
    std::shared_ptr<const mesh::Mesh> mesh = _function_space->mesh();
    assert(mesh);
    const std::size_t gdim = mesh->geometry().dim();
    const std::size_t tdim = mesh->topology().dim();
    auto map = mesh->topology().index_map(tdim);

    // Get geometry data
    const graph::AdjacencyList<std::int32_t>& x_dofmap
        = mesh->geometry().dofmap();
    const std::size_t num_dofs_g = mesh->geometry().cmap().dim();
    std::span<const double> x_g = mesh->geometry().x();

    // Get coordinate map
    const CoordinateElement& cmap = mesh->geometry().cmap();

    // Get element
    std::shared_ptr<const FiniteElement> element = _function_space->element();
    assert(element);
    const int bs_element = element->block_size();
    const std::size_t reference_value_size
        = element->reference_value_size() / bs_element;
    const std::size_t value_size = element->value_size() / bs_element;
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
    std::vector<T> coefficients(space_dimension * bs_element);

    // Get dofmap
    std::shared_ptr<const DofMap> dofmap = _function_space->dofmap();
    assert(dofmap);
    const int bs_dof = dofmap->bs();

    std::span<const std::uint32_t> cell_info;
    if (element->needs_dof_transformations())
    {
      mesh->topology_mutable().create_entity_permutations();
      cell_info = std::span(mesh->topology().get_cell_permutation_info());
    }

    namespace stdex = std::experimental;
    using cmdspan4_t
        = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;
    using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
    using mdspan3_t = stdex::mdspan<double, stdex::dextents<std::size_t, 3>>;

    std::vector<double> coord_dofs_b(num_dofs_g * gdim);
    mdspan2_t coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);
    std::vector<double> xp_b(1 * gdim);
    mdspan2_t xp(xp_b.data(), 1, gdim);

    // Loop over points
    std::fill(u.data(), u.data() + u.size(), 0.0);
    std::span<const T> _v = _x->array();

    // Evaluate geometry basis at point (0, 0, 0) on the reference cell.
    // Used in affine case.
    std::array<std::size_t, 4> phi0_shape = cmap.tabulate_shape(1, 1);
    std::vector<double> phi0_b(std::reduce(phi0_shape.begin(), phi0_shape.end(),
                                           1, std::multiplies{}));
    cmdspan4_t phi0(phi0_b.data(), phi0_shape);
    cmap.tabulate(1, std::vector<double>(tdim), {1, tdim}, phi0_b);
    auto dphi0 = stdex::submdspan(phi0, std::pair(1, tdim + 1), 0,
                                  stdex::full_extent, 0);

    // Data structure for evaluating geometry basis at specific points.
    // Used in non-affine case.
    std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, 1);
    std::vector<double> phi_b(
        std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
    cmdspan4_t phi(phi_b.data(), phi_shape);
    auto dphi = stdex::submdspan(phi, std::pair(1, tdim + 1), 0,
                                 stdex::full_extent, 0);

    // Reference coordinates for each point
    std::vector<double> Xb(xshape[0] * tdim);
    mdspan2_t X(Xb.data(), xshape[0], tdim);

    // Geometry data at each point
    std::vector<double> J_b(xshape[0] * gdim * tdim);
    mdspan3_t J(J_b.data(), xshape[0], gdim, tdim);
    std::vector<double> K_b(xshape[0] * tdim * gdim);
    mdspan3_t K(K_b.data(), xshape[0], tdim, gdim);
    std::vector<double> detJ(xshape[0]);
    std::vector<double> det_scratch(2 * gdim * tdim);

    // Prepare geometry data in each cell
    for (std::size_t p = 0; p < cells.size(); ++p)
    {
      const int cell_index = cells[p];

      // Skip negative cell indices
      if (cell_index < 0)
        continue;

      // Get cell geometry (coordinate dofs)
      auto x_dofs = x_dofmap.links(cell_index);
      assert(x_dofs.size() == num_dofs_g);
      for (std::size_t i = 0; i < num_dofs_g; ++i)
      {
        const int pos = 3 * x_dofs[i];
        for (std::size_t j = 0; j < gdim; ++j)
          coord_dofs(i, j) = x_g[pos + j];
      }

      for (std::size_t j = 0; j < gdim; ++j)
        xp(0, j) = x[p * xshape[1] + j];

      auto _J = stdex::submdspan(J, p, stdex::full_extent, stdex::full_extent);
      auto _K = stdex::submdspan(K, p, stdex::full_extent, stdex::full_extent);

      std::array<double, 3> Xpb = {0, 0, 0};
      stdex::mdspan<double,
                    stdex::extents<std::size_t, 1, stdex::dynamic_extent>>
          Xp(Xpb.data(), 1, tdim);

      // Compute reference coordinates X, and J, detJ and K
      if (cmap.is_affine())
      {
        CoordinateElement::compute_jacobian(dphi0, coord_dofs, _J);
        CoordinateElement::compute_jacobian_inverse(_J, _K);
        std::array<double, 3> x0 = {0, 0, 0};
        for (std::size_t i = 0; i < coord_dofs.extent(1); ++i)
          x0[i] += coord_dofs(0, i);
        CoordinateElement::pull_back_affine(Xp, _K, x0, xp);
        detJ[p]
            = CoordinateElement::compute_jacobian_determinant(_J, det_scratch);
      }
      else
      {
        // Pull-back physical point xp to reference coordinate Xp
        cmap.pull_back_nonaffine(Xp, xp, coord_dofs);

        cmap.tabulate(1, std::span(Xpb.data(), tdim), {1, tdim}, phi_b);
        CoordinateElement::compute_jacobian(dphi, coord_dofs, _J);
        CoordinateElement::compute_jacobian_inverse(_J, _K);
        detJ[p]
            = CoordinateElement::compute_jacobian_determinant(_J, det_scratch);
      }

      for (std::size_t j = 0; j < X.extent(1); ++j)
        X(p, j) = Xpb[j];
    }

    // Prepare basis function data structures
    std::vector<double> basis_derivatives_reference_values_b(
        1 * xshape[0] * space_dimension * reference_value_size);
    cmdspan4_t basis_derivatives_reference_values(
        basis_derivatives_reference_values_b.data(), 1, xshape[0],
        space_dimension, reference_value_size);
    std::vector<double> basis_values_b(space_dimension * value_size);
    mdspan2_t basis_values(basis_values_b.data(), space_dimension, value_size);

    // Compute basis on reference element
    element->tabulate(basis_derivatives_reference_values_b, Xb,
                      {X.extent(0), X.extent(1)}, 0);

    using xu_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
    using xU_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
    using xJ_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
    using xK_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
    auto push_forward_fn
        = element->basix_element().map_fn<xu_t, xU_t, xJ_t, xK_t>();

    auto apply_dof_transformation
        = element->get_dof_transformation_function<double>();
    const std::size_t num_basis_values = space_dimension * reference_value_size;

    for (std::size_t p = 0; p < cells.size(); ++p)
    {
      const int cell_index = cells[p];

      // Skip negative cell indices
      if (cell_index < 0)
        continue;

      // Permute the reference values to account for the cell's
      // orientation
      apply_dof_transformation(
          std::span(basis_derivatives_reference_values_b.data()
                        + p * num_basis_values,
                    num_basis_values),
          cell_info, cell_index, reference_value_size);

      {
        auto _U = stdex::submdspan(basis_derivatives_reference_values, 0, p,
                                   stdex::full_extent, stdex::full_extent);
        auto _J
            = stdex::submdspan(J, p, stdex::full_extent, stdex::full_extent);
        auto _K
            = stdex::submdspan(K, p, stdex::full_extent, stdex::full_extent);
        push_forward_fn(basis_values, _U, _J, detJ[p], _K);
      }

      // Get degrees of freedom for current cell
      std::span<const std::int32_t> dofs = dofmap->cell_dofs(cell_index);
      for (std::size_t i = 0; i < dofs.size(); ++i)
        for (int k = 0; k < bs_dof; ++k)
          coefficients[bs_dof * i + k] = _v[bs_dof * dofs[i] + k];

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

  /// Name
  std::string name = "u";

private:
  // The function space
  std::shared_ptr<const FunctionSpace> _function_space;

  // The vector of expansion coefficients (local)
  std::shared_ptr<la::Vector<T>> _x;
};
} // namespace dolfinx::fem
