// Copyright (C) 2007-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <petscsys.h>
#include <vector>

namespace dolfin
{

namespace function
{
class Function;
class FunctionSpace;
} // namespace function

namespace mesh
{
class Mesh;
} // namespace mesh

namespace fem
{

/// Marking function to define facets when DirichletBC applies
using marking_function = std::function<Eigen::Array<bool, Eigen::Dynamic, 1>(
    const Eigen::Ref<
        const Eigen::Array<double, 3, Eigen::Dynamic, Eigen::RowMajor>>&)>;

/// Locate degrees of freedom topologically
///
/// Finds degrees of freedom which belong to mesh entities. This doesn't
/// work elements will degrees-of-freedom that are associated with cell
/// interiors, e.g. for discontinuous elements.
///
/// @param[in] V The function (sub)space on which degrees of freedom
///              will be located
/// @param[in] entity_dim Topological dimension of mesh entities on
///                       which degrees of freedom will be located
/// @param[in] entities Indices of mesh entities. All dofs associated
///                     with these indices will be returned.
/// @return Array of local indices of located degrees of freedom
Eigen::Array<PetscInt, Eigen::Dynamic, 1> locate_dofs_topological(
    const function::FunctionSpace& V, const int entity_dim,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>&
        entities);

/// Locate degrees of freedom geometrically
///
/// Finds degrees of freedom whose geometric coordinate is true for a
/// marking function.
///
/// @param[in] V The function (sub)space on which degrees of freedom
///              will be located
/// @param[in] marker Function marking tabulated degrees of freedom
/// @return Array of local indices of located degrees of freedom
Eigen::Array<PetscInt, Eigen::Dynamic, 1>
locate_dofs_geometrical(const function::FunctionSpace& V,
                        marking_function marker);

/// Interface for setting (strong) Dirichlet boundary conditions.
///
///     u = g on G,
///
/// where u is the solution to be computed, g is a function and G is a
/// sub domain of the mesh.
///
/// A DirichletBC is specified by the function g, the function space
/// (trial space) and degrees of freedom to which the boundary condition
/// applies.
///

class DirichletBC
{

public:
  /// Create boundary condition
  ///
  /// @param[in] V The function (sub)space on which the boundary
  ///              condition is applied
  /// @param[in] g The boundary condition value
  /// @param[in] V_dofs Degree-of-freedom indices in the space where the
  ///                   the boundary condition is applied
  /// @param[in] g_dofs Degree-of-freedom indices in the space of the
  ///                   boundary value function
  DirichletBC(
      std::shared_ptr<const function::FunctionSpace> V,
      std::shared_ptr<const function::Function> g,
      const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& V_dofs,
      const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>&
          g_dofs);

  /// Copy constructor
  /// @param[in] bc The object to be copied
  DirichletBC(const DirichletBC& bc) = default;

  /// Move constructor
  /// @param[in] bc The object to be moved
  DirichletBC(DirichletBC&& bc) = default;

  /// Destructor
  ~DirichletBC() = default;

  /// Assignment operator
  /// @param[in] bc Another DirichletBC object
  DirichletBC& operator=(const DirichletBC& bc) = default;

  /// Move assignment operator
  DirichletBC& operator=(DirichletBC&& bc) = default;

  /// Return function space to which boundary conditions are applied
  /// @return The function space
  std::shared_ptr<const function::FunctionSpace> function_space() const;

  /// Return boundary value function g
  /// @return The boundary values Function
  std::shared_ptr<const function::Function> value() const;

  /// Get array of dof indices to which a Dirichlet boundary condition
  /// is applied. The array is sorted and may contain ghost entries.
  Eigen::Array<PetscInt, Eigen::Dynamic, 2>& dofs();

  /// Get array of dof indices owned by this process to which a
  /// Dirichlet BC is applied. The array is sorted and does not contain
  /// ghost entries.
  const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 2>>
  dofs_owned() const;

  // FIXME: clarify w.r.t ghosts
  /// Set bc entries in x to scale*x_bc
  void set(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x,
           double scale = 1.0) const;

  // FIXME: clarify w.r.t ghosts
  /// Set bc entries in x to scale*(x0 - x_bc).
  void set(
      Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x,
      const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>& x0,
      double scale = 1.0) const;

  // FIXME: clarify  w.r.t ghosts

  /// Set boundary condition value for entres with an applied boundary
  /// condition. Other entries are not modified.
  void dof_values(
      Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> values) const;

  // FIXME: clarify w.r.t ghosts
  /// Set markers[i] = true if dof i has a boundary condition applied
  /// Value of markers[i] is not changed otherwise
  void mark_dofs(std::vector<bool>& markers) const;

private:
  // The function space (possibly a sub function space)
  std::shared_ptr<const function::FunctionSpace> _function_space;

  // The function
  std::shared_ptr<const function::Function> _g;

  // Indices of dofs in _function_space and in the space of _g
  Eigen::Array<PetscInt, Eigen::Dynamic, 2> _dofs;

  // The first _owned_indices in _dofs are owned by this process
  int _owned_indices = -1;
};
} // namespace fem
} // namespace dolfin
