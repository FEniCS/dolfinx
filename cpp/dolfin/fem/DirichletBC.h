// Copyright (C) 2007-2018 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
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

/// Interface for setting (strong) Dirichlet boundary conditions.
///
///     u = g on G,
///
/// where u is the solution to be computed, g is a function and G is a
/// sub domain of the mesh.
///
/// A DirichletBC is specified by the function g, the function space
/// (trial space) and boundary indicators on (a subset of) the mesh
/// boundary.
///
/// The boundary indicators may be specified in a number of different
/// ways:
///
/// 1. Providing a marking function, mark(x, only_boundary), to
///    specify on which facets the boundary conditions should be
///    applied.
/// 2. Providing list of facets (by index, local to a process).
///
/// The degrees-of-freedom to which boundary conditions are applied are
/// computed at construction and cannot be changed afterwards.
///
/// The 'method' variable may be used to specify the type of method used
/// to identify degrees of freedom on the boundary. Available methods
/// are:
///
/// 1. topological approach (default)
///
///    Fastest, but will only identify degrees of freedom that are
///    located on a facet that is entirely on the / boundary. In
///    particular, the topological approach will not identify degrees of
///    freedom for discontinuous elements (which are all internal to the
///    cell).
///
/// 2. (not yet implemented) geometric approach
///
///    Each dof on each facet that matches the boundary condition will
///    be checked.
///
/// 3. (not yet implemented) pointwise approach.
///
///    For pointwise boundary conditions e.g. pointloads..

class DirichletBC
{

public:
  /// Method of boundary condition application
  enum class Method
  {
    topological,
    geometric,
    pointwise
  };

  /// Create boundary condition with marking method
  ///
  /// @param[in] V (FunctionSpace)
  ///         The function space
  /// @param[in] g (Function)
  ///         The value
  /// @param[in] mark (std::function)
  ///         The marking method
  /// @param[in] method (std::string)
  ///         Optional argument: A string specifying
  ///         the method to identify dofs
  /// @param[in] check_midpoint (bool)
  DirichletBC(std::shared_ptr<const function::FunctionSpace> V,
              std::shared_ptr<const function::Function> g,
              const std::function<Eigen::Array<bool, Eigen::Dynamic, 1>(
                  const Eigen::Ref<const Eigen::Array<
                      double, Eigen::Dynamic, 3, Eigen::RowMajor>>&)>& marker,
              Method method = Method::topological);

  /// Create boundary condition with facet indices
  ///
  /// @param[in] V (FunctionSpace)
  ///         The function space.
  /// @param[in] g (Function)
  ///         The value.
  /// @param[in] markers (std::vector<std:size_t>&)
  ///         Boundary markers (facet index local to process)
  /// @param[in] method (std::string)
  ///         Optional argument: A string specifying the
  ///         method to identify dofs.
  DirichletBC(std::shared_ptr<const function::FunctionSpace> V,
              std::shared_ptr<const function::Function> g,
              const std::vector<std::int32_t>& facet_indices,
              Method method = Method::topological);

  /// Copy constructor. Either cached DOF data are copied.
  ///
  /// @param[in] bc (DirichletBC&)
  ///         The object to be copied.
  DirichletBC(const DirichletBC& bc) = default;

  /// Move constructor
  DirichletBC(DirichletBC&& bc) = default;

  /// Destructor
  ~DirichletBC() = default;

  /// Assignment operator. Either cached DOF data are assigned.
  ///
  /// @param[in] bc (DirichletBC)
  ///         Another DirichletBC object.
  DirichletBC& operator=(const DirichletBC& bc) = default;

  /// Move assignment operator
  DirichletBC& operator=(DirichletBC&& bc) = default;

  /// Return function space V
  ///
  /// @return FunctionSpace
  ///         The function space to which boundary conditions are applied.
  std::shared_ptr<const function::FunctionSpace> function_space() const;

  /// Return boundary value g
  ///
  /// @return Function
  ///         The boundary values Function. Returns null if it does not
  ///         exist.
  std::shared_ptr<const function::Function> value() const;

  // FIXME: clarify  w.r.t ghosts
  // FIXME: clarify length of returned array
  /// Get array of dof indices to which a Dirichlet BC is applied. The
  /// array is sorted.
  const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>
  dof_indices() const;

  // FIXME: clarify w.r.t ghosts
  /// Set bc entries in x to scale*x_bc
  void set(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x,
           double scale = 1.0) const;

  // FIXME: clarify w.r.t ghosts
  /// Set bc entries in x to scale*(x0 - x_bc).
  void set(
      Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x,
      const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x0,
      double scale = 1.0) const;

  // FIXME: clarify  w.r.t ghosts
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

  // Vector tuples (dof in _function_space, dof in g-space) to which bcs
  // are applied, i.e. u[dofs[i][0]] = g[dofs[i][1]] where u is in
  // _function_space.
  Eigen::Array<PetscInt, Eigen::Dynamic, 2, Eigen::RowMajor> _dofs;

  // Indices in _function_space to which bcs are applied. Must be sorted.
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> _dof_indices;
};
} // namespace fem
} // namespace dolfin
