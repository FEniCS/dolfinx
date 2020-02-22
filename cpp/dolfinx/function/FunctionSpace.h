// Copyright (C) 2008-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <dolfinx/fem/FiniteElement.h>
#include <functional>
#include <map>
#include <memory>
#include <petscsys.h>
#include <vector>

namespace dolfinx
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

  /// Return global dimension of the function space
  /// @return The dimension of the function space
  std::int64_t dim() const;

  /// Interpolate a finite element Function into this function space,
  /// filling the array of expansion coefficients associated with this
  /// function space
  /// @param[in,out] coefficients The expansion coefficients. It must be
  ///                             correctly sized by the calling
  ///                             function.
  /// @param[in] v The function to be interpolated
  void interpolate(
      Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> coefficients,
      const Function& v) const;

  /// Interpolation function
  using interpolation_function = std::function<void(
      Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>>,
      const Eigen::Ref<
          const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>&)>;

  /// Interpolate an expression into this function space, filling the
  /// array of expansion coefficients associated with this function
  /// space.
  /// @cond Work around doxygen bug for std::function
  /// @param[in,out] coefficients The expansion coefficients. It must be
  ///                             correctly sized by the calling
  ///                             function.
  /// @param[in] f The function to be interpolated
  /// @endcond
  void interpolate(
      Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> coefficients,
      const std::function<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                       Eigen::Dynamic, Eigen::RowMajor>(
          const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                              Eigen::RowMajor>>&)>& f) const;

  /// Interpolate an expression into this function space, filling the
  /// array of expansion coefficients associated with this function
  /// space.
  /// @note This interface is not intended for general use. It supports
  ///       the use of an expression function with a C-signature; it is
  ///       typically used by compiled Numba functions with C interface.
  /// @param[in,out] coefficients The expansion coefficients to be
  ///                             filled. It must be correctly sized by
  ///                             the calling function.
  /// @param[in] f The function to be interpolated
  void interpolate_c(
      Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> coefficients,
      const interpolation_function& f) const;

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
  /// @return The new function space and a map rom new to old dofs
  std::pair<std::shared_ptr<FunctionSpace>, std::vector<std::int32_t>>
  collapse() const;

  /// Check if function space has given element
  /// @param[in] element The finite element
  /// @return  True if the function space has the given element
  bool has_element(const fem::FiniteElement& element) const
  {
    return element.hash() == this->_element->hash();
  }

  /// Get the component with respect to the root superspace
  /// @return The component with respect to the root superspace , i.e.
  ///         W.sub(1).sub(0) == [1, 0]
  std::vector<int> component() const;

  /// Tabulate the physical coordinates of all dofs on this process.
  /// @return The dof coordinates [([x0, y0, z0], [x1, y1, z1], ...)
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>
  tabulate_dof_coordinates() const;

  /// Set dof entries in vector to value*x[i], where [x][i] is the
  /// coordinate of the dof spatial coordinate. Parallel layout of
  /// vector must be consistent with dof map range This function is
  /// typically used to construct the null space of a matrix operator,
  /// e.g. rigid body rotations.
  ///
  /// @param[in,out] x The vector to set
  /// @param[in] value The value to multiply to coordinate by
  /// @param[in] component The coordinate index
  void set_x(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> x,
             PetscScalar value, int component) const;

  /// Unique identifier
  std::size_t id() const;

  /// The mesh
  std::shared_ptr<const mesh::Mesh> mesh() const;

  /// The finite element
  std::shared_ptr<const fem::FiniteElement> element() const;

  /// The dofmap
  std::shared_ptr<const fem::DofMap> dofmap() const;

private:
  // Interpolate data. Fills coefficients using 'values', which are the
  // values of an expression at each dof.
  void interpolate(
      Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> coefficients,
      const Eigen::Ref<const Eigen::Array<PetscScalar, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>&
          values) const;

  // The mesh
  std::shared_ptr<const mesh::Mesh> _mesh;

  // The finite element
  std::shared_ptr<const fem::FiniteElement> _element;

  // The dofmap
  std::shared_ptr<const fem::DofMap> _dofmap;

  // General interpolation from any Function on any mesh
  void interpolate_from_any(
      Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>>
          expansion_coefficients,
      const Function& v) const;

  // The component w.r.t. to root space
  std::vector<int> _component;

  // Unique identifier
  std::size_t _id;

  // The identifier of root space
  std::size_t _root_space_id;

  // Cache of subspaces
  mutable std::map<std::vector<int>, std::weak_ptr<FunctionSpace>> _subspaces;
};
} // namespace function
} // namespace dolfinx
