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

class Function;
class GenericDofMap;
class GenericFunction;
class PETScVector;
class Mesh;

/// This class represents a finite element function space defined by
/// a mesh, a finite element, and a local-to-global mapping of the
/// degrees of freedom (dofmap).

class FunctionSpace : public Variable
{
public:
  /// Create function space for given mesh, element and dofmap
  /// (shared data)
  ///
  /// *Arguments*
  ///     mesh (_Mesh_)
  ///         The mesh.
  ///     element (_FiniteElement_)
  ///         The element.
  ///     dofmap (_GenericDofMap_)
  ///         The dofmap.
  FunctionSpace(std::shared_ptr<const Mesh> mesh,
                std::shared_ptr<const FiniteElement> element,
                std::shared_ptr<const GenericDofMap> dofmap);

protected:
  /// Create empty function space for later initialization. This
  /// constructor is intended for use by any sub-classes which need
  /// to construct objects before the initialisation of the base
  /// class. Data can be attached to the base class using
  /// FunctionSpace::attach(...).
  ///
  /// *Arguments*
  ///     mesh (_Mesh_)
  ///         The mesh.
  explicit FunctionSpace(std::shared_ptr<const Mesh> mesh);

public:
  /// Copy constructor
  ///
  /// *Arguments*
  ///     V (_FunctionSpace_)
  ///         The object to be copied.
  FunctionSpace(const FunctionSpace& V);

  /// Destructor
  virtual ~FunctionSpace();

protected:
  /// Attach data to an empty function space
  ///
  /// *Arguments*
  ///     element (_FiniteElement_)
  ///         The element.
  ///     dofmap (_GenericDofMap_)
  ///         The dofmap.
  void attach(std::shared_ptr<const FiniteElement> element,
              std::shared_ptr<const GenericDofMap> dofmap);

public:
  /// Assignment operator
  ///
  /// *Arguments*
  ///     V (_FunctionSpace_)
  ///         Another function space.
  const FunctionSpace& operator=(const FunctionSpace& V);

  /// Equality operator
  ///
  /// *Arguments*
  ///     V (_FunctionSpace_)
  ///         Another function space.
  bool operator==(const FunctionSpace& V) const;

  /// Inequality operator
  ///
  /// *Arguments*
  ///     V (_FunctionSpace_)
  ///         Another function space.
  bool operator!=(const FunctionSpace& V) const;

  /// Return mesh
  ///
  /// *Returns*
  ///     _Mesh_
  ///         The mesh.
  std::shared_ptr<const Mesh> mesh() const;

  /// Return finite element
  ///
  /// *Returns*
  ///     _FiniteElement_
  ///         The finite element.
  std::shared_ptr<const FiniteElement> element() const;

  /// Return dofmap
  ///
  /// *Returns*
  ///     _GenericDofMap_
  ///         The dofmap.
  std::shared_ptr<const GenericDofMap> dofmap() const;

  /// Return global dimension of the function space.
  /// Equivalent to dofmap()->global_dimension()
  ///
  /// *Returns*
  ///     std::size_t
  ///         The dimension of the function space.
  std::int64_t dim() const;

  /// Interpolate function v into function space, returning the
  /// vector of expansion coefficients
  ///
  /// *Arguments*
  ///     expansion_coefficients (_PETScVector_)
  ///         The expansion coefficients.
  ///     v (_GenericFunction_)
  ///         The function to be interpolated.
  void interpolate(PETScVector& expansion_coefficients,
                   const GenericFunction& v) const;

  /// Extract subspace for component
  ///
  /// *Arguments*
  ///     component (std::vector<std::size_t>)
  ///         The component.
  ///
  /// *Returns*
  ///     _FunctionSpace_
  ///         The subspace.
  std::shared_ptr<FunctionSpace>
  sub(const std::vector<std::size_t>& component) const;

  /// Check whether V is subspace of this, or this itself
  ///
  /// *Arguments*
  ///     V (_FunctionSpace_)
  ///         The space to be tested for inclusion.
  ///
  /// *Returns*
  ///     bool
  ///         True if V is contained or equal to this.
  bool contains(const FunctionSpace& V) const;

  /// Collapse a subspace and return a new function space
  ///
  /// *Returns*
  ///     _FunctionSpace_
  ///         The new function space.
  std::shared_ptr<FunctionSpace> collapse() const;

  /// Collapse a subspace and return a new function space and a map
  /// from new to old dofs
  ///
  /// *Arguments*
  ///     collapsed_dofs (std::unordered_map<std::size_t, std::size_t>)
  ///         The map from new to old dofs.
  ///
  /// *Returns*
  ///     _FunctionSpace_
  ///       The new function space.
  std::shared_ptr<FunctionSpace>
  collapse(std::unordered_map<std::size_t, std::size_t>& collapsed_dofs) const;

  /// Check if function space has given cell
  ///
  /// *Arguments*
  ///     cell (_Cell_)
  ///         The cell.
  ///
  /// *Returns*
  ///     bool
  ///         True if the function space has the given cell.
  bool has_cell(const Cell& cell) const { return &cell.mesh() == &(*_mesh); }

  /// Check if function space has given element
  ///
  /// *Arguments*
  ///     element (_FiniteElement_)
  ///         The finite element.
  ///
  /// *Returns*
  ///     bool
  ///         True if the function space has the given element.
  bool has_element(const FiniteElement& element) const
  {
    return element.hash() == _element->hash();
  }

  /// Return component w.r.t. to root superspace, i.e.
  ///   W.sub(1).sub(0) == [1, 0].
  ///
  /// *Returns*
  ///     std::vector<std::size_t>
  ///         The component (w.r.t to root superspace).
  std::vector<std::size_t> component() const;

  /// Tabulate the coordinates of all dofs on this process. This
  /// function is typically used by preconditioners that require the
  /// spatial coordinates of dofs, for example for re-partitioning or
  /// nullspace computations.
  ///
  /// *Arguments*
  ///     mesh (_Mesh_)
  ///         The mesh.
  ///
  /// *Returns*
  ///     std::vector<double>
  ///         The dof coordinates (x0, y0, x1, y1, . . .)
  std::vector<double> tabulate_dof_coordinates() const;

  /// Set dof entries in vector to value*x[i], where [x][i] is the
  /// coordinate of the dof spatial coordinate. Parallel layout of
  /// vector must be consistent with dof map range This function is
  /// typically used to construct the null space of a matrix
  /// operator, e.g. rigid body rotations.
  ///
  /// *Arguments*
  ///     vector (_PETScVector_)
  ///         The vector to set.
  ///     value (double)
  ///         The value to multiply to coordinate by.
  ///     component (std::size_t)
  ///         The coordinate index.
  ///     mesh (_Mesh_)
  ///         The mesh.
  void set_x(PETScVector& x, double value, std::size_t component) const;

  /// Return informal string representation (pretty-print)
  ///
  /// *Arguments*
  ///     verbose (bool)
  ///         Flag to turn on additional output.
  ///
  /// *Returns*
  ///     std::string
  ///         An informal representation of the function space.
  std::string str(bool verbose) const;

  /// Print dofmap (useful for debugging)
  void print_dofmap() const;

private:
  // General interpolation from any GenericFunction on any mesh
  void interpolate_from_any(PETScVector& expansion_coefficients,
                            const GenericFunction& v) const;

  // The mesh
  std::shared_ptr<const Mesh> _mesh;

  // The finite element
  std::shared_ptr<const FiniteElement> _element;

  // The dofmap
  std::shared_ptr<const GenericDofMap> _dofmap;

  // The component w.r.t. to root space
  std::vector<std::size_t> _component;

  // The identifier of root space
  std::size_t _root_space_id;

  // Cache of subspaces
  mutable std::map<std::vector<std::size_t>, std::weak_ptr<FunctionSpace>>
      _subspaces;
};
}


