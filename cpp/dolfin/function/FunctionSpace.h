// Copyright (C) 2008-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/mesh/Cell.h>
#include <functional>
#include <map>
#include <memory>
#include <petscsys.h>
#include <vector>

namespace dolfin
{

namespace fem
{
class DofMap;
}

namespace mesh
{
class Mesh;
}

namespace function
{
class Function;

/// This class represents a finite element function space defined by
/// a mesh, a finite element, and a local-to-global mapping of the
/// degrees of freedom (dofmap).

class FunctionSpace
{
public:
  /// Create function space for given mesh, element and dofmap
  /// (shared data)
  ///
  /// @param    mesh (_mesh::Mesh_)
  ///         The mesh.
  /// @param    element (_FiniteElement_)
  ///         The element.
  /// @param    dofmap (_DofMap_)
  ///         The dofmap.
  FunctionSpace(std::shared_ptr<const mesh::Mesh> mesh,
                std::shared_ptr<const fem::FiniteElement> element,
                std::shared_ptr<const fem::DofMap> dofmap);

  // Copy constructor (deleted)
  FunctionSpace(const FunctionSpace& V) = delete;

  /// Move constructor
  FunctionSpace(FunctionSpace&& V) = default;

  /// Destructor
  virtual ~FunctionSpace() = default;

  // Assignment operator (delete)
  FunctionSpace& operator=(const FunctionSpace& V) = delete;

  // Move assignment operator (delete)
  FunctionSpace& operator=(FunctionSpace&& V) = default;

  /// Equality operator
  ///
  /// @param    V (_FunctionSpace_)
  ///         Another function space.
  bool operator==(const FunctionSpace& V) const;

  /// Inequality operator
  ///
  /// @param   V (_FunctionSpace_)
  ///         Another function space.
  bool operator!=(const FunctionSpace& V) const;

  /// Return mesh
  ///
  /// @returns  _mesh::Mesh_
  ///         The mesh.
  std::shared_ptr<const mesh::Mesh> mesh() const;

  /// Return finite element
  ///
  /// @returns _FiniteElement_
  ///         The finite element.
  std::shared_ptr<const fem::FiniteElement> element() const;

  /// Return dofmap
  ///
  /// @returns _DofMap_
  ///         The dofmap.
  std::shared_ptr<const fem::DofMap> dofmap() const;

  /// Return global dimension of the function space.
  /// Equivalent to dofmap()->global_dimension()
  ///
  /// @returns    std::size_t
  ///         The dimension of the function space.
  std::int64_t dim() const;

  /// Interpolate function v into function space, returning the
  /// vector of expansion coefficients
  ///
  /// @param   expansion_coefficients
  ///         The expansion coefficients.
  /// @param    v (_Function_)
  ///         The function to be interpolated.
  void interpolate(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>
                       expansion_coefficients,
                   const Function& v) const;

  /// Interpolate expression into function space, returning the
  /// vector of expansion coefficients
  ///
  /// @param   expansion_coefficients (_la::PETScVector_)
  ///         The expansion coefficients.
  /// @param   expr (_Expression_)
  ///         The expression to be interpolated.
  void interpolate(
      Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>
          expansion_coefficients,
      const std::function<void(
          Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>,
          const Eigen::Ref<const Eigen::Array<
              double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>)>& f)
      const;

  /// Extract subspace for component
  ///
  /// @param    component (std::vector<std::size_t>)
  ///         The component.
  ///
  /// @returns    _FunctionSpace_
  ///         The subspace.
  std::shared_ptr<FunctionSpace> sub(const std::vector<int>& component) const;

  /// Check whether V is subspace of this, or this itself
  ///
  /// @param    V (_FunctionSpace_)
  ///         The space to be tested for inclusion.
  ///
  /// @returns    bool
  ///         True if V is contained or equal to this.
  bool contains(const FunctionSpace& V) const;

  /// Collapse a subspace and return a new function space and a map
  /// from new to old dofs
  ///
  /// @param    collapsed_dofs (std::vector<PetscInt>)
  ///         The map from new to old dofs.
  ///
  /// @returns    _FunctionSpace_
  ///       The new function space.
  std::pair<std::shared_ptr<FunctionSpace>, std::vector<PetscInt>>
  collapse() const;

  /// Check if function space has given cell
  ///
  /// @param    cell (_Cell_)
  ///         The cell.
  ///
  /// @returns    bool
  ///         True if the function space has the given cell.
  bool has_cell(const mesh::Cell& cell) const
  {
    return &cell.mesh() == &(*_mesh);
  }

  /// Check if function space has given element
  ///
  /// @param    element (_FiniteElement_)
  ///         The finite element.
  ///
  /// @returns    bool
  ///         True if the function space has the given element.
  bool has_element(const fem::FiniteElement& element) const
  {
    return element.hash() == _element->hash();
  }

  /// Return component w.r.t. to root superspace, i.e.
  ///   W.sub(1).sub(0) == [1, 0].
  ///
  /// @returns   std::vector<int>
  ///         The component (w.r.t to root superspace).
  std::vector<int> component() const;

  /// Tabulate the coordinates of all dofs on this process. This
  /// function is typically used by preconditioners that require the
  /// spatial coordinates of dofs, for example for re-partitioning or
  /// nullspace computations.
  ///
  /// @returns    EigenRowArrayXXd
  ///         The dof coordinates [([0, y0], [x1, y1], . . .)
  EigenRowArrayXXd tabulate_dof_coordinates() const;

  /// Set dof entries in vector to value*x[i], where [x][i] is the
  /// coordinate of the dof spatial coordinate. Parallel layout of
  /// vector must be consistent with dof map range This function is
  /// typically used to construct the null space of a matrix operator,
  /// e.g. rigid body rotations.
  ///
  /// @param x
  ///         The vector to set.
  /// @param value (double)
  ///         The value to multiply to coordinate by.
  /// @param component (int)
  ///         The coordinate index.
  void set_x(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x,
             PetscScalar value, int component) const;

  /// Return informal string representation (pretty-print)
  ///
  /// @param    verbose (bool)
  ///         Flag to turn on additional output.
  ///
  /// @returns    std::string
  ///         An informal representation of the function space.
  std::string str(bool verbose) const;

  /// Print dofmap (useful for debugging)
  void print_dofmap() const;

  /// Unique identifier
  const std::size_t id;

private:
  // General interpolation from any Function on any mesh
  void interpolate_from_any(
      Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>
          expansion_coefficients,
      const Function& v) const;

  // The mesh
  std::shared_ptr<const mesh::Mesh> _mesh;

  // The finite element
  std::shared_ptr<const fem::FiniteElement> _element;

  // The dofmap
  std::shared_ptr<const fem::DofMap> _dofmap;

  // The component w.r.t. to root space
  std::vector<int> _component;

  // The identifier of root space
  std::size_t _root_space_id;

  // Cache of subspaces
  mutable std::map<std::vector<int>, std::weak_ptr<FunctionSpace>> _subspaces;
};
} // namespace function
} // namespace dolfin
