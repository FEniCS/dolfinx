// Copyright (C) 2003-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "FunctionSpace.h"
#include "interpolate.h"
#include <Eigen/Dense>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/UniqueIdGenerator.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <functional>
#include <memory>
#include <petscvec.h>
#include <string>
#include <vector>

namespace dolfinx::function
{

class FunctionSpace;

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
  /// Create function on given function space
  /// @param[in] V The function space
  explicit Function(std::shared_ptr<const FunctionSpace> V)
      : _id(common::UniqueIdGenerator::id()), _function_space(V),
        _x(std::make_shared<la::Vector<T>>(V->dofmap()->index_map))
  {
    if (!V->component().empty())
    {
      throw std::runtime_error("Cannot create Function from subspace. Consider "
                               "collapsing the function space");
    }
    _x->array().setZero();
  }

  /// Create function on given function space with a given vector
  ///
  /// *Warning: This constructor is intended for internal library use only*
  ///
  /// @param[in] V The function space
  /// @param[in] x The vector
  Function(std::shared_ptr<const FunctionSpace> V,
           std::shared_ptr<la::Vector<T>> x)
      : _id(common::UniqueIdGenerator::id()), _function_space(V), _x(x)
  {
    // We do not check for a subspace since this constructor is used for
    // creating subfunctions

    // Assertion uses '<=' to deal with sub-functions
    assert(V->dofmap());
    assert(V->dofmap()->index_map->size_global()
               * V->dofmap()->index_map->block_size()
           <= _x->map()->block_size() * _x->map()->size_global());
  }

  // Copy constructor
  Function(const Function& v) = delete;

  /// Move constructor
  Function(Function&& v) = default;

  /// Destructor
  virtual ~Function()
  {
    if (_petsc_vector)
      VecDestroy(&_petsc_vector);
  }

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

  /// Collapse a subfunction (view into the Function) to a stand-alone
  /// Function
  /// @return New collapsed Function
  Function collapse() const
  {
    // Create new collapsed FunctionSpace
    const auto [function_space_new, collapsed_map]
        = _function_space->collapse();

    // Create new vector
    assert(function_space_new);
    auto vector_new = std::make_shared<la::Vector<T>>(
        function_space_new->dofmap()->index_map);

    // Copy values into new vector
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& x_old = _x->array();
    Eigen::Matrix<T, Eigen::Dynamic, 1>& x_new = vector_new->array();
    for (std::size_t i = 0; i < collapsed_map.size(); ++i)
    {
      assert((int)i < x_new.size());
      assert(collapsed_map[i] < x_old.size());
      x_new[i] = x_old[collapsed_map[i]];
    }

    return Function(function_space_new, vector_new);
  }

  /// Return shared pointer to function space
  /// @return The function space
  std::shared_ptr<const FunctionSpace> function_space() const
  {
    return _function_space;
  }

  /// Return vector of expansion coefficients as a PETSc Vec. Throws an
  /// exception a PETSc Vec cannot be created due to a type mismatch.
  /// @return The vector of expansion coefficients
  Vec vector() const
  {
    // Check that this is not a sub function
    assert(_function_space->dofmap());
    assert(_function_space->dofmap()->index_map);
    if (_x->map()->block_size() * _x->map()->size_global()
        != _function_space->dofmap()->index_map->size_global()
               * _function_space->dofmap()->index_map->block_size())
    {
      throw std::runtime_error(
          "Cannot access a non-const vector from a subfunction");
    }

    // Check that data type is the same as the PETSc build
    if constexpr (std::is_same<T, PetscScalar>::value)
    {
      if (!_petsc_vector)
      {
        _petsc_vector = la::create_ghosted_vector(
            *_function_space->dofmap()->index_map, _x->array());
      }
      return _petsc_vector;
    }
    else
    {
      throw std::runtime_error(
          "Cannot return PETSc vector wrapper. Type mismatch");
    }
  }

  /// Underlying vector
  std::shared_ptr<const la::Vector<T>> x() const { return _x; }

  /// Underlying vector
  std::shared_ptr<la::Vector<T>> x() { return _x; }

  /// Interpolate a Function (on possibly non-matching meshes)
  /// @param[in] v The function to be interpolated.
  void interpolate(const Function<T>& v) { function::interpolate(*this, v); }

  /// Interpolate an expression
  /// @param[in] f The expression to be interpolated
  void
  interpolate(const std::function<
              Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(
                  const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                                      Eigen::RowMajor>>&)>& f)
  {
    function::interpolate(*this, f);
  }

  /// Evaluate the Function at points
  ///
  /// @param[in] x The coordinates of the points. It has shape
  ///   (num_points, 3).
  /// @param[in] cells An array of cell indices. cells[i] is the index
  ///   of the cell that contains the point x(i). Negative cell indices
  ///   can be passed, and the corresponding point will be ignored.
  /// @param[in,out] u The values at the points. Values are not computed
  ///   for points with a negative cell index. This argument must be
  ///   passed with the corrext size.
  void
  eval(const Eigen::Ref<
           const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>& x,
       const Eigen::Ref<const Eigen::Array<int, Eigen::Dynamic, 1>>& cells,
       Eigen::Ref<
           Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
           u) const
  {
    // TODO: This could be easily made more efficient by exploiting points
    // being ordered by the cell to which they belong.

    if (x.rows() != cells.rows())
    {
      throw std::runtime_error(
          "Number of points and number of cells must be equal.");
    }
    if (x.rows() != u.rows())
    {
      throw std::runtime_error(
          "Length of array for Function values must be the "
          "same as the number of points.");
    }

    // Get mesh
    assert(_function_space);
    std::shared_ptr<const mesh::Mesh> mesh = _function_space->mesh();
    assert(mesh);
    const int gdim = mesh->geometry().dim();
    const int tdim = mesh->topology().dim();

    // Get geometry data
    const graph::AdjacencyList<std::int32_t>& x_dofmap
        = mesh->geometry().dofmap();

    // FIXME: Add proper interface for num coordinate dofs
    const int num_dofs_g = x_dofmap.num_links(0);
    const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
        = mesh->geometry().x();
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        coordinate_dofs(num_dofs_g, gdim);

    // Get coordinate map
    const fem::CoordinateElement& cmap = mesh->geometry().cmap();

    // Get element
    assert(_function_space->element());
    std::shared_ptr<const fem::FiniteElement> element
        = _function_space->element();
    assert(element);
    const int block_size = element->block_size();
    const int reference_value_size
        = element->reference_value_size() / block_size;
    const int value_size = element->value_size() / block_size;
    const int space_dimension = element->space_dimension() / block_size;

    // Prepare geometry data structures
    Eigen::Tensor<double, 3, Eigen::RowMajor> J(1, gdim, tdim);
    Eigen::Array<double, Eigen::Dynamic, 1> detJ(1);
    Eigen::Tensor<double, 3, Eigen::RowMajor> K(1, tdim, gdim);
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(
        1, tdim);

    // Prepare basis function data structures
    Eigen::Tensor<double, 3, Eigen::RowMajor> basis_reference_values(
        1, space_dimension, reference_value_size);
    Eigen::Tensor<double, 3, Eigen::RowMajor> basis_values(1, space_dimension,
                                                           value_size);

    // Create work vector for expansion coefficients
    Eigen::Matrix<T, 1, Eigen::Dynamic> coefficients(space_dimension
                                                     * block_size);

    // Get dofmap
    std::shared_ptr<const fem::DofMap> dofmap = _function_space->dofmap();
    assert(dofmap);

    mesh->topology_mutable().create_entity_permutations();
    const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
        = mesh->topology().get_cell_permutation_info();

    // Loop over points
    u.setZero();
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& _v = _x->array();
    for (Eigen::Index p = 0; p < cells.rows(); ++p)
    {
      const int cell_index = cells(p);

      // Skip negative cell indices
      if (cell_index < 0)
        continue;

      // Get cell geometry (coordinate dofs)
      auto x_dofs = x_dofmap.links(cell_index);
      for (int i = 0; i < num_dofs_g; ++i)
        coordinate_dofs.row(i) = x_g.row(x_dofs[i]).head(gdim);

      // Compute reference coordinates X, and J, detJ and K
      cmap.compute_reference_geometry(X, J, detJ, K, x.row(p).head(gdim),
                                      coordinate_dofs);

      // Compute basis on reference element
      element->evaluate_reference_basis(basis_reference_values, X);

      // Push basis forward to physical element
      element->transform_reference_basis(basis_values, basis_reference_values,
                                         X, J, detJ, K, cell_info[cell_index]);

      // Get degrees of freedom for current cell
      auto dofs = dofmap->cell_dofs(cell_index);
      for (Eigen::Index i = 0; i < dofs.size(); ++i)
        coefficients[i] = _v[dofs[i]];

      // Compute expansion
      for (int block = 0; block < block_size; ++block)
      {
        for (int i = 0; i < space_dimension; ++i)
        {
          for (int j = 0; j < value_size; ++j)
          {
            // TODO: Find an Eigen shortcut for this operation
            u.row(p)[j * block_size + block]
                += coefficients[i * block_size + block] * basis_values(0, i, j);
          }
        }
      }
    }
  }

  /// Compute values at all mesh 'nodes'
  /// @return The values at all geometric points
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  compute_point_values() const
  {
    assert(_function_space);
    std::shared_ptr<const mesh::Mesh> mesh = _function_space->mesh();
    assert(mesh);
    const int tdim = mesh->topology().dim();

    // Compute in tensor (one for scalar function, . . .)
    const int value_size_loc = _function_space->element()->value_size();

    // Resize Array for holding point values
    Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        point_values(mesh->geometry().x().rows(), value_size_loc);

    // Prepare cell geometry
    const graph::AdjacencyList<std::int32_t>& x_dofmap
        = mesh->geometry().dofmap();

    // FIXME: Add proper interface for num coordinate dofs
    const int num_dofs_g = x_dofmap.num_links(0);
    const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
        = mesh->geometry().x();

    // Interpolate point values on each cell (using last computed value if
    // not continuous, e.g. discontinuous Galerkin methods)
    Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> x(num_dofs_g, 3);
    Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> values(
        num_dofs_g, value_size_loc);
    auto map = mesh->topology().index_map(tdim);
    assert(map);
    const std::int32_t num_cells = map->size_local() + map->num_ghosts();
    for (std::int32_t c = 0; c < num_cells; ++c)
    {
      // Get coordinates for all points in cell
      auto dofs = x_dofmap.links(c);
      for (int i = 0; i < num_dofs_g; ++i)
        x.row(i) = x_g.row(dofs[i]);

      values.resize(x.rows(), value_size_loc);

      // Call evaluate function
      Eigen::Array<int, Eigen::Dynamic, 1> cells(x.rows());
      cells = c;
      eval(x, cells, values);

      // Copy values to array of point values
      for (int i = 0; i < x.rows(); ++i)
        point_values.row(dofs[i]) = values.row(i);
    }

    return point_values;
  }

  /// Name
  std::string name = "u";

  /// ID
  std::size_t id() const { return _id; }

private:
  // ID
  std::size_t _id;

  // The function space
  std::shared_ptr<const FunctionSpace> _function_space;

  // The vector of expansion coefficients (local)
  std::shared_ptr<la::Vector<T>> _x;

  // PETSc wrapper of the expansion coefficients
  mutable Vec _petsc_vector = nullptr;
};
} // namespace dolfinx::function
