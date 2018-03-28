// Copyright (C) 2008-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include <dolfin/common/Variable.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/mesh/Cell.h>

namespace dolfin
{
namespace la
{
class PETScVector;
}

namespace fem
{
class GenericDofMap;
}

namespace mesh
{
class Mesh;
}

namespace function
{
class Function;
class GenericFunction;

/// This class represents a finite element function space defined by
/// a mesh, a finite element, and a local-to-global mapping of the
/// degrees of freedom (dofmap).

class FunctionSpace : public common::Variable
{
public:
  /// Create function space for given mesh, element and dofmap
  /// (shared data)
  ///
  /// @param    mesh (_mesh::Mesh_)
  ///         The mesh.
  /// @param    element (_FiniteElement_)
  ///         The element.
  /// @param    dofmap (_GenericDofMap_)
  ///         The dofmap.
  FunctionSpace(std::shared_ptr<const mesh::Mesh> mesh,
                std::shared_ptr<const fem::FiniteElement> element,
                std::shared_ptr<const fem::GenericDofMap> dofmap);

protected:
  /// Create empty function space for later initialization. This
  /// constructor is intended for use by any sub-classes which need
  /// to construct objects before the initialisation of the base
  /// class. Data can be attached to the base class using
  /// FunctionSpace::attach(...).
  ///
  /// @param    mesh (_mesh::Mesh_)
  ///         The mesh.
  explicit FunctionSpace(std::shared_ptr<const mesh::Mesh> mesh);

public:
  /// Copy constructor
  ///
  /// @param    V (_FunctionSpace_)
  ///         The object to be copied.
  FunctionSpace(const FunctionSpace& V);

  /// Destructor
  virtual ~FunctionSpace();

protected:
  /// Attach data to an empty function space
  ///
  /// @param    element (_FiniteElement_)
  ///         The element.
  /// @param    dofmap (_GenericDofMap_)
  ///         The dofmap.
  void attach(std::shared_ptr<const fem::FiniteElement> element,
              std::shared_ptr<const fem::GenericDofMap> dofmap);

public:
  /// Assignment operator
  ///
  /// @param    V (_FunctionSpace_)
  ///         Another function space.
  const FunctionSpace& operator=(const FunctionSpace& V);

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
  /// @returns _GenericDofMap_
  ///         The dofmap.
  std::shared_ptr<const fem::GenericDofMap> dofmap() const;

  /// Return global dimension of the function space.
  /// Equivalent to dofmap()->global_dimension()
  ///
  /// @returns    std::size_t
  ///         The dimension of the function space.
  std::int64_t dim() const;

  /// Interpolate function v into function space, returning the
  /// vector of expansion coefficients
  ///
  /// @param   expansion_coefficients (_la::PETScVector_)
  ///         The expansion coefficients.
  /// @param    v (_GenericFunction_)
  ///         The function to be interpolated.
  void interpolate(la::PETScVector& expansion_coefficients,
                   const GenericFunction& v) const;

  /// Extract subspace for component
  ///
  /// @param    component (std::vector<std::size_t>)
  ///         The component.
  ///
  /// @returns    _FunctionSpace_
  ///         The subspace.
  std::shared_ptr<FunctionSpace>
  sub(const std::vector<std::size_t>& component) const;

  /// Check whether V is subspace of this, or this itself
  ///
  /// @param    V (_FunctionSpace_)
  ///         The space to be tested for inclusion.
  ///
  /// @returns    bool
  ///         True if V is contained or equal to this.
  bool contains(const FunctionSpace& V) const;

  /// Collapse a subspace and return a new function space
  ///
  /// @returns    _FunctionSpace_
  ///         The new function space.
  std::shared_ptr<FunctionSpace> collapse() const;

  /// Collapse a subspace and return a new function space and a map
  /// from new to old dofs
  ///
  /// @param    collapsed_dofs (std::unordered_map<std::size_t, std::size_t>)
  ///         The map from new to old dofs.
  ///
  /// @returns    _FunctionSpace_
  ///       The new function space.
  std::shared_ptr<FunctionSpace>
  collapse(std::unordered_map<std::size_t, std::size_t>& collapsed_dofs) const;

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
  /// @returns   std::vector<std::size_t>
  ///         The component (w.r.t to root superspace).
  std::vector<std::size_t> component() const;

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
  /// typically used to construct the null space of a matrix
  /// operator, e.g. rigid body rotations.
  ///
  /// @param x (_la::PETScVector_)
  ///         The vector to set.
  /// @param value (double)
  ///         The value to multiply to coordinate by.
  /// @param component (std::size_t)
  ///         The coordinate index.
  void set_x(la::PETScVector& x, double value, std::size_t component) const;

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

private:
  // General interpolation from any GenericFunction on any mesh
  void interpolate_from_any(la::PETScVector& expansion_coefficients,
                            const GenericFunction& v) const;

  // The mesh
  std::shared_ptr<const mesh::Mesh> _mesh;

  // The finite element
  std::shared_ptr<const fem::FiniteElement> _element;

  // The dofmap
  std::shared_ptr<const fem::GenericDofMap> _dofmap;

  // The component w.r.t. to root space
  std::vector<std::size_t> _component;

  // The identifier of root space
  std::size_t _root_space_id;

  // Cache of subspaces
  mutable std::map<std::vector<std::size_t>, std::weak_ptr<FunctionSpace>>
      _subspaces;
};
}
}
