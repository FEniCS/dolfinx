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
#include <string>
#include <utility>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>
#include <xtl/xspan.hpp>

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
    xtl::span<const T> x_old = _x->array();
    xtl::span<T> x_new = x->mutable_array();
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
  void interpolate(const Function<T>& v,
                   const xtl::span<const std::int32_t>& cells)
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
      const std::function<xt::xarray<T>(const xt::xtensor<double, 2>&)>& f,
      const xtl::span<const std::int32_t>& cells)
  {
    assert(_function_space);
    assert(_function_space->element());
    assert(_function_space->mesh());
    const xt::xtensor<double, 2> x = fem::interpolation_coords(
        *_function_space->element(), *_function_space->mesh(), cells);
    auto fx = f(x);
    if (int vs = _function_space->element()->value_size();
        vs == 1 and fx.dimension() == 1)
    {
      // Check for scalar-valued functions
      if (fx.shape(0) != x.shape(1))
        throw std::runtime_error("Data returned by callable has wrong length");
    }
    else
    {
      // Check for vector/tensor value
      if (fx.dimension() != 2)
        throw std::runtime_error("Expected 2D array of data");
      if (fx.shape(0) != vs)
      {
        throw std::runtime_error(
            "Data returned by callable has wrong shape(0) size");
      }
      if (fx.shape(1) != x.shape(1))
      {
        throw std::runtime_error(
            "Data returned by callable has wrong shape(1) size");
      }
    }

    fem::interpolate(*this, fx, cells);
  }

  /// Interpolate an expression function on the whole domain
  /// @param[in] f The expression to be interpolated
  void interpolate(
      const std::function<xt::xarray<T>(const xt::xtensor<double, 2>&)>& f)
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
  void interpolate(const Expression<T>& e,
                   const xtl::span<const std::int32_t>& cells)
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

    if (!xt::allclose(e.X(),
                      _function_space->element()->interpolation_points()))
    {
      throw std::runtime_error("Function element interpolation points not "
                               "equal to Expression interpolation points");
    }

    // Array to hold evaluted Expression
    std::size_t num_cells = cells.size();
    std::size_t num_points = e.X().shape(0);
    xt::xtensor<T, 3> f({num_cells, num_points, value_size});

    // Evaluate Expression at points
    auto f_view = xt::reshape_view(f, {num_cells, num_points * value_size});
    e.eval(cells, f_view);

    // Reshape evaluated data to fit interpolate
    // Expression returns matrix of shape (num_cells, num_points *
    // value_size), i.e. xyzxyz ordering of dof values per cell per point.
    // The interpolation uses xxyyzz input, ordered for all points of each
    // cell, i.e. (value_size, num_cells*num_points)
    xt::xarray<T> _f = xt::reshape_view(xt::transpose(f, {2, 0, 1}),
                                        {value_size, num_cells * num_points});

    // Interpolate values into appropriate space
    fem::interpolate(*this, _f, cells);
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

  /// Evaluate the Function at points
  ///
  /// @param[in] x The coordinates of the points. It has shape
  /// (num_points, 3).
  /// @param[in] cells An array of cell indices. cells[i] is the index
  /// of the cell that contains the point x(i). Negative cell indices
  /// can be passed, and the corresponding point will be ignored.
  /// @param[in,out] u The values at the points. Values are not computed
  /// for points with a negative cell index. This argument must be
  /// passed with the correct size.
  void eval(const xt::xtensor<double, 2>& x,
            const xtl::span<const std::int32_t>& cells,
            xt::xtensor<T, 2>& u) const
  {
    // TODO: This could be easily made more efficient by exploiting points
    // being ordered by the cell to which they belong.

    if (x.shape(0) != cells.size())
    {
      throw std::runtime_error(
          "Number of points and number of cells must be equal.");
    }
    if (x.shape(0) != u.shape(0))
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
    // FIXME: Add proper interface for num coordinate dofs
    const std::size_t num_dofs_g = x_dofmap.num_links(0);
    xtl::span<const double> x_g = mesh->geometry().x();

    // Get coordinate map
    const fem::CoordinateElement& cmap = mesh->geometry().cmap();

    // Get element
    assert(_function_space->element());
    std::shared_ptr<const fem::FiniteElement> element
        = _function_space->element();
    assert(element);
    const int bs_element = element->block_size();
    const std::size_t reference_value_size
        = element->reference_value_size() / bs_element;
    const std::size_t value_size = element->value_size() / bs_element;
    const std::size_t space_dimension = element->space_dimension() / bs_element;

    // If the space has sub elements, concatenate the evaluations on the sub
    // elements
    const int num_sub_elements = element->num_sub_elements();
    if (num_sub_elements > 1 and num_sub_elements != bs_element)
    {
      throw std::runtime_error("Function::eval is not supported for mixed "
                               "elements. Extract subspaces.");
    }

    // Create work vector for expansion coefficients
    std::vector<T> coefficients(space_dimension * bs_element);

    // Get dofmap
    std::shared_ptr<const fem::DofMap> dofmap = _function_space->dofmap();
    assert(dofmap);
    const int bs_dof = dofmap->bs();

    xtl::span<const std::uint32_t> cell_info;
    if (element->needs_dof_transformations())
    {
      mesh->topology_mutable().create_entity_permutations();
      cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
    }

    xt::xtensor<double, 2> coordinate_dofs
        = xt::zeros<double>({num_dofs_g, gdim});
    xt::xtensor<double, 2> xp = xt::zeros<double>({std::size_t(1), gdim});

    // Loop over points
    std::fill(u.data(), u.data() + u.size(), 0.0);
    const xtl::span<const T>& _v = _x->array();

    // -- Lambda function for affine pull-backs
    xt::xtensor<double, 4> data(cmap.tabulate_shape(1, 1));
    const xt::xtensor<double, 2> X0(xt::zeros<double>({std::size_t(1), tdim}));
    cmap.tabulate(1, X0, data);
    const xt::xtensor<double, 2> dphi_i
        = xt::view(data, xt::range(1, tdim + 1), 0, xt::all(), 0);
    auto pull_back_affine = [dphi_i](auto&& X, const auto& cell_geometry,
                                     auto&& J, auto&& K, const auto& x) mutable
    {
      fem::CoordinateElement::compute_jacobian(dphi_i, cell_geometry, J);
      fem::CoordinateElement::compute_jacobian_inverse(J, K);
      fem::CoordinateElement::pull_back_affine(
          X, K, fem::CoordinateElement::x0(cell_geometry), x);
    };

    xt::xtensor<double, 2> dphi;
    xt::xtensor<double, 2> X({x.shape(0), tdim});
    xt::xtensor<double, 3> J = xt::zeros<double>({x.shape(0), gdim, tdim});
    xt::xtensor<double, 3> K = xt::zeros<double>({x.shape(0), tdim, gdim});
    xt::xtensor<double, 1> detJ = xt::zeros<double>({x.shape(0)});
    xt::xtensor<double, 4> phi(cmap.tabulate_shape(1, 1));

    xt::xtensor<double, 2> _Xp({1, tdim});
    for (std::size_t p = 0; p < cells.size(); ++p)
    {
      const int cell_index = cells[p];
      // Skip negative cell indices
      if (cell_index < 0)
        continue;

      // Get cell geometry (coordinate dofs)
      auto x_dofs = x_dofmap.links(cell_index);
      for (std::size_t i = 0; i < num_dofs_g; ++i)
      {
        const int pos = 3 * x_dofs[i];
        for (std::size_t j = 0; j < gdim; ++j)
          coordinate_dofs(i, j) = x_g[pos + j];
      }

      for (std::size_t j = 0; j < gdim; ++j)
        xp(0, j) = x(p, j);

      auto _J = xt::view(J, p, xt::all(), xt::all());
      auto _K = xt::view(K, p, xt::all(), xt::all());
      // Compute reference coordinates X, and J, detJ and K
      if (cmap.is_affine())
      {
        pull_back_affine(_Xp, coordinate_dofs, _J, _K, xp);
        detJ[p]
            = dolfinx::fem::CoordinateElement::compute_jacobian_determinant(_J);
      }
      else
      {
        cmap.pull_back_nonaffine(_Xp, xp, coordinate_dofs);
        cmap.tabulate(1, _Xp, phi);
        dphi = xt::view(phi, xt::range(1, tdim + 1), 0, xt::all(), 0);
        dolfinx::fem::CoordinateElement::compute_jacobian(dphi, coordinate_dofs,
                                                          _J);
        dolfinx::fem::CoordinateElement::compute_jacobian_inverse(_J, _K);
        detJ[p]
            = dolfinx::fem::CoordinateElement::compute_jacobian_determinant(_J);
      }
      xt::row(X, p) = xt::row(_Xp, 0);
    }

    // Prepare basis function data structures
    xt::xtensor<double, 4> basis_derivatives_reference_values(
        {1, x.shape(0), space_dimension, reference_value_size});
    xt::xtensor<double, 3> basis_values(
        {static_cast<std::size_t>(1), space_dimension, value_size});

    // Compute basis on reference element
    element->tabulate(basis_derivatives_reference_values, X, 0);

    using u_t = xt::xview<decltype(basis_values)&, std::size_t,
                          xt::xall<std::size_t>, xt::xall<std::size_t>>;
    using U_t
        = xt::xview<decltype(basis_derivatives_reference_values)&, std::size_t,
                    std::size_t, xt::xall<std::size_t>, xt::xall<std::size_t>>;
    using J_t = xt::xview<decltype(J)&, std::size_t, xt::xall<std::size_t>,
                          xt::xall<std::size_t>>;
    using K_t = xt::xview<decltype(K)&, std::size_t, xt::xall<std::size_t>,
                          xt::xall<std::size_t>>;
    auto push_forward_fn = element->map_fn<u_t, U_t, J_t, K_t>();
    const std::function<void(const xtl::span<double>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>
        apply_dof_transformation
        = element->get_dof_transformation_function<double>();
    const std::size_t num_basis_values = space_dimension * reference_value_size;

    for (std::size_t p = 0; p < cells.size(); ++p)
    {
      const int cell_index = cells[p];
      // Skip negative cell indices
      if (cell_index < 0)
        continue;

      // Permute the reference values to account for the cell's orientation
      apply_dof_transformation(
          xtl::span(basis_derivatives_reference_values.data()
                        + p * num_basis_values,
                    num_basis_values),
          cell_info, cell_index, reference_value_size);

      auto _K = xt::view(K, p, xt::all(), xt::all());
      auto _J = xt::view(J, p, xt::all(), xt::all());
      auto _U = xt::view(basis_derivatives_reference_values, (std::size_t)0, p,
                         xt::all(), xt::all());
      auto _u = xt::view(basis_values, (std::size_t)0, xt::all(), xt::all());
      push_forward_fn(_u, _U, _J, detJ[p], _K);

      // Get degrees of freedom for current cell
      xtl::span<const std::int32_t> dofs = dofmap->cell_dofs(cell_index);
      for (std::size_t i = 0; i < dofs.size(); ++i)
        for (int k = 0; k < bs_dof; ++k)
          coefficients[bs_dof * i + k] = _v[bs_dof * dofs[i] + k];

      // Compute expansion
      auto u_row = xt::row(u, p);
      for (int k = 0; k < bs_element; ++k)
      {
        for (std::size_t i = 0; i < space_dimension; ++i)
        {
          for (std::size_t j = 0; j < value_size; ++j)
          {
            u_row[j * bs_element + k]
                += coefficients[bs_element * i + k] * basis_values(0, i, j);
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
