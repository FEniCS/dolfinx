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

/// Locate degrees of freedom topologically
///
/// Finds degrees of freedom which belong to mesh entities. This
/// doesn't work e.g. for discontinuous function spaces.
///
/// @param[in] V The function (sub)space on which degrees of freedom will
///              be located
/// @param[in] entity_dim Topological dimension of mesh entities on which
///                       degrees of freedom will be located
/// @param[in] entities Indices of mesh entities on which degrees of freedom
///                     will be located
/// @return Array of local indices of located degrees of freedom
///
std::vector<PetscInt>
locate_dofs_topological(const function::FunctionSpace& V, const int entity_dim,
                        const std::vector<std::int32_t>& entities);

/// Locate degrees of freedom geometrically
///
/// Finds degrees of freedom whose location satisfies a marking function.
///
/// @param[in] V The function (sub)space on which degrees of freedom will
///              be located
/// @param[in] marker Function marking tabulated degrees of freedom
/// @return Array of local indices of located degrees of freedom
///
Eigen::Array<PetscInt, Eigen::Dynamic, 1> locate_dofs_geometrical(
    const function::FunctionSpace& V,
    std::function<Eigen::Array<bool, Eigen::Dynamic, 1>(
        const Eigen::Ref<
            const Eigen::Array<double, 3, Eigen::Dynamic, Eigen::RowMajor>>&)> marker);

/// Interface for setting (strong) Dirichlet boundary conditions.
///
///     u = g on G,
///
/// where u is the solution to be computed, g is a function and G is a
/// sub domain of the mesh.
///
/// A DirichletBC is specified by the function g, the function space
/// (trial space) and degrees of freedom to which the boundary condition
///  applies.
///

class DirichletBC
{

public:

  /// Marking function to define facets when DirichletBC applies
  using marking_function = std::function<Eigen::Array<bool, Eigen::Dynamic, 1>(
      const Eigen::Ref<
          const Eigen::Array<double, 3, Eigen::Dynamic, Eigen::RowMajor>>&)>;

  /// Create boundary condition
  ///
  /// @param[in] V The function (sub)space on which the boundary
  ///              condition is applied
  /// @param[in] g The boundary condition value
  /// @param[in] V_dofs Indices of degrees of freedom in the space where the
  ///                   the boundary condition is applied
  /// @param[in] g_dofs Indices of degrees of freedom in the space of the
  ///                   boundary value function
  DirichletBC(
      std::shared_ptr<const function::FunctionSpace> V,
      std::shared_ptr<const function::Function> g,
      const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& V_dofs,
      const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>&
          g_dofs);

  /// Copy constructor. Either cached DOF data are copied.
  /// @param[in] bc The object to be copied.
  DirichletBC(const DirichletBC& bc) = default;

  /// Move constructor
  DirichletBC(DirichletBC&& bc) = default;

  /// Destructor
  ~DirichletBC() = default;

  /// Assignment operator. Either cached DOF data are assigned.
  /// @param[in] bc Another DirichletBC object.
  DirichletBC& operator=(const DirichletBC& bc) = default;

  /// Move assignment operator
  DirichletBC& operator=(DirichletBC&& bc) = default;

  /// Return function space V
  /// @return The function space to which boundary conditions are
  ///          applied.
  std::shared_ptr<const function::FunctionSpace> function_space() const;

  /// Return boundary value g
  /// @return The boundary values Function. Returns null if it does not
  ///         exist.
  std::shared_ptr<const function::Function> value() const;

  /// Get array of dof indices to which a Dirichlet BC is applied. The
  /// array is sorted and may contain ghost entries.
  Eigen::Array<PetscInt, Eigen::Dynamic, 2>& dofs();

  /// Get array of dof indices owned by this process to which a
  /// Dirichlet BC is applied. The array is sorted and does not contain ghost
  /// entries.
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

  // FIXME: clarify  w.r.t ghostss
  /// Set boundary condition value for entres with an applied boundary
  /// condition. Other entries are not modified.
  void dof_values(
      Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> values) const;

  // FIXME: clarify w.r.t ghosts
  /// Set markers[i] = true if dof i has a boundary condition applied.
  /// Value of markers[i] is not changed otherwise.
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
