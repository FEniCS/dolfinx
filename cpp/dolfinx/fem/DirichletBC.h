// Copyright (C) 2007-2020 Michal Habera, Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <petscsys.h>
#include <vector>

namespace dolfinx
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

/// Build an array of degree-of-freedom indices that are associated with
/// give mesh entities (topological)
///
/// Finds degrees-of-freedom which belong to provided mesh entities.
/// Note that degrees-of-freedom for discontinuous elements are
/// associated with the cell even if they may appear to be associated
/// with a facet/edge/vertex.
///
/// @param[in] V The function (sub)space(s) on which degrees-of-freedom
///     (DOFs) will be located. The spaces must share the same mesh and
///     element type.
/// @param[in] dim Topological dimension of mesh entities on
///     which degrees-of-freedom will be located
/// @param[in] entities Indices of mesh entities. All DOFs associated
///     with the closure of these indices will be returned
/// @param[in] remote True to return also "remotely located"
///     degree-of-freedom indices. Remotely located degree-of-freedom
///     indices are local/owned by the current process, but which the
///     current process cannot identify because it does not recognize
///     mesh entity as a marked. For example, a boundary condition dof
///     at a vertex where this process does not have the associated
///     boundary facet. This commonly occurs with partitioned meshes.
/// @return Array of local DOF indices in the spaces V[0] (and V[1] is
///     two spaces are passed in). If two spaces are passed in, the (i,
///     0) entry is the DOF index in the space V[0] and (i, 1) is the
///     correspinding DOF entry in the space V[1].
Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic>
locate_dofs_topological(
    const std::vector<std::reference_wrapper<function::FunctionSpace>>& V,
    const int dim, const Eigen::Ref<const Eigen::ArrayXi>& entities,
    bool remote = true);

/// Build an array of degree-of-freedom indices based on coordinates of
/// the degree-of-freedom (geometric).
///
/// Finds degrees of freedom whose geometric coordinate is true for the
/// provided marking function.
///
/// @attention This function is slower than the topological version
///
/// @param[in] V The function (sub)space(s) on which degrees of freedom
///     will be located. The spaces must share the same mesh and
///     element type.
/// @param[in] marker Function marking tabulated degrees of freedom
/// @return Array of local DOF indices in the spaces V[0] (and V[1] is
///     two spaces are passed in). If two spaces are passed in, the (i,
///     0) entry is the DOF index in the space V[0] and (i, 1) is the
///     correspinding DOF entry in the space V[1].
Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic>
locate_dofs_geometrical(
    const std::vector<std::reference_wrapper<function::FunctionSpace>>& V,
    marking_function marker);

/// Interface for setting (strong) Dirichlet boundary conditions
///
///     u = g on G,
///
/// where u is the solution to be computed, g is a function and G is a
/// sub domain of the mesh.
///
/// A DirichletBC is specified by the function g, the function space
/// (trial space) and degrees of freedom to which the boundary condition
/// applies.

class DirichletBC
{

public:
  /// Create boundary condition
  ///
  /// @param[in] g The boundary condition value. The boundary condition
  /// can be applied to a a function on the same space as g.
  /// @param[in] dofs Degree-of-freedom indices in the space of the
  ///   boundary value function applied to V_dofs[i]
  DirichletBC(
      std::shared_ptr<const function::Function> g,
      const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>&
          dofs);

  /// Create boundary condition
  ///
  /// @param[in] g The boundary condition value
  /// @param[in] V_g_dofs 2D array of degree-of-freedom indices. First
  ///   column are indices in the space where boundary condition is
  ///   applied (V), second column are indices in the space of the
  ///   boundary condition value function g.
  /// @param[in] V The function (sub)space on which the boundary
  ///   condition is applied
  DirichletBC(
      std::shared_ptr<const function::Function> g,
      const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 2>>&
          V_g_dofs,
      std::shared_ptr<const function::FunctionSpace> V);

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

  /// The function space to which boundary conditions are applied
  /// @return The function space
  std::shared_ptr<const function::FunctionSpace> function_space() const;

  /// Return boundary value function g
  /// @return The boundary values Function
  std::shared_ptr<const function::Function> value() const;

  /// Get array of dof indices to which a Dirichlet boundary condition
  /// is applied. The array is sorted and may contain ghost entries.
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 2>& dofs() const;

  /// Get array of dof indices owned by this process to which a
  /// Dirichlet BC is applied. The array is sorted and does not contain
  /// ghost entries.
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 2>>
  dofs_owned() const;

  /// Set bc entries in x to scale*x_bc
  /// @todo Clarify w.r.t ghosts
  void set(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x,
           double scale = 1.0) const;

  /// Set bc entries in x to scale*(x0 - x_bc).
  /// @todo Clarify w.r.t ghosts
  void
  set(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x,
      const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>& x0,
      double scale = 1.0) const;

  /// Set boundary condition value for entres with an applied boundary
  /// condition. Other entries are not modified.
  /// @todo Clarify w.r.t ghosts
  void dof_values(
      Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> values) const;

  /// Set markers[i] = true if dof i has a boundary condition applied.
  /// Value of markers[i] is not changed otherwise.
  /// @todo Clarify w.r.t ghosts
  void mark_dofs(std::vector<bool>& markers) const;

private:
  // The function space (possibly a sub function space)
  std::shared_ptr<const function::FunctionSpace> _function_space;

  // The function
  std::shared_ptr<const function::Function> _g;

  // Indices of dofs in _function_space and in the space of _g
  Eigen::Array<std::int32_t, Eigen::Dynamic, 2> _dofs;

  // The first _owned_indices in _dofs are owned by this process
  int _owned_indices = -1;
};
} // namespace fem
} // namespace dolfinx
