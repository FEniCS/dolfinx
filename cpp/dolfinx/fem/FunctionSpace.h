// Copyright (C) 2008-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <vector>
#include <xtensor/xtensor.hpp>

namespace dolfinx::mesh
{
class Mesh;
}

namespace dolfinx::fem
{
class DofMap;
class FiniteElement;

/// This class represents a finite element function space defined by a
/// mesh, a finite element, and a local-to-global map of the degrees of
/// freedom (dofmap).

class FunctionSpace
{
public:
  /// Create function space for given mesh, element and dofmap
  /// @param[in] mesh The mesh
  /// @param[in] element The element
  /// @param[in] dofmap The dofmap
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

  /// Move assignment operator
  FunctionSpace& operator=(FunctionSpace&& V) = default;

  /// Equality operator
  /// @param[in] V Another function space
  bool operator==(const FunctionSpace& V) const;

  /// Inequality operator
  /// @param[in] V Another function space.
  bool operator!=(const FunctionSpace& V) const;

  /// Extract subspace for component
  /// @param[in] component The subspace component
  /// @return The subspace
  std::shared_ptr<FunctionSpace> sub(const std::vector<int>& component) const;

  /// Check whether V is subspace of this, or this itself
  /// @param[in] V The space to be tested for inclusion
  /// @return True if V is contained in or equal to this FunctionSpace
  bool contains(const FunctionSpace& V) const;

  /// Collapse a subspace and return a new function space and a map from
  /// new to old dofs
  /// @return The new function space and a map from new to old dofs
  std::pair<FunctionSpace, std::vector<std::int32_t>> collapse() const;

  /// Get the component with respect to the root superspace
  /// @return The component with respect to the root superspace , i.e.
  ///         W.sub(1).sub(0) == [1, 0]
  std::vector<int> component() const;

  /// @todo Remove - see function in interpolate.h
  /// Tabulate the physical coordinates of all dofs on this process.
  /// @param[in] transpose If false the returned data has shape
  /// (num_points, gdim), otherwise it is transposed and has shape
  /// (gdim, num_points)
  /// @return The dof coordinates [([x0, y0, z0], [x1, y1, z1], ...) if
  /// @p transpose is false, and otherwise the returned data is
  /// transposed.
  xt::xtensor<double, 2> tabulate_dof_coordinates(bool transpose) const;

  /// Unique identifier
  std::size_t id() const;

  /// The mesh
  std::shared_ptr<const mesh::Mesh> mesh() const;

  /// The finite element
  std::shared_ptr<const fem::FiniteElement> element() const;

  /// The dofmap
  std::shared_ptr<const fem::DofMap> dofmap() const;

private:
  // The mesh
  std::shared_ptr<const mesh::Mesh> _mesh;

  // The finite element
  std::shared_ptr<const fem::FiniteElement> _element;

  // The dofmap
  std::shared_ptr<const fem::DofMap> _dofmap;

  // The component w.r.t. to root space
  std::vector<int> _component;

  // Unique identifier
  std::size_t _id;

  // The identifier of root space
  std::size_t _root_space_id;

  // Cache of subspaces
  mutable std::map<std::vector<int>, std::weak_ptr<FunctionSpace>> _subspaces;
};

/// Extract FunctionSpaces for (0) rows blocks and (1) columns blocks
/// from a rectangular array of (test, trial) space pairs. The test
/// space must be the same for each row and the trial spaces must be the
/// same for each column. Raises an exception if there is an
/// inconsistency. e.g. if each form in row i does not have the same
/// test space then an exception is raised.
///
/// @param[in] V Vector function spaces for (0) each row block and (1)
/// each column block
std::array<std::vector<std::shared_ptr<const FunctionSpace>>, 2>
common_function_spaces(
    const std::vector<
        std::vector<std::array<std::shared_ptr<const FunctionSpace>, 2>>>& V);
} // namespace dolfinx::fem
