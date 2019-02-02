// Copyright (C) 2007-2018 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <map>
#include <memory>
#include <petscsys.h>
#include <set>
#include <utility>
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
class SubDomain;
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
/// 1. Providing a_SubDomain_ object, using the inside() function to
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
/// 2. geometric approach
///
///    Each dof on each facet that matches the boundary condition will
///    be checked.
///
/// 3. pointwise approach.
///
///    For pointwise boundary conditions e.g. pointloads..
///
///    Note: when using "pointwise", the boolean argument `on_boundary`
///    in SubDomain::inside will always be false.
///
/// The 'check_midpoint' variable can be used to decide whether or not
/// the midpoint of each facet should be checked when a user-defined
/// _SubDomain_ is used to define the domain of the boundary condition.
/// By default, midpoints are always checked. Note that this variable
/// may be of importance close to corners, in which case it is sometimes
/// important to check the midpoint to avoid including facets "on the
/// diagonal close" to a corner. This variable is also of importance for
/// curved boundaries (like on a sphere or cylinder), in which case it
/// is important *not* to check the midpoint which will be located in
/// the interior of a domain defined relative to a radius.

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

  /// Create boundary condition for subdomain
  ///
  /// @param[in] V (FunctionSpace)
  ///         The function space
  /// @param[in] g (Function)
  ///         The value
  /// @param[in] sub_domain (mesh::SubDomain)
  ///         The subdomain
  /// @param[in] method (std::string)
  ///         Optional argument: A string specifying
  ///         the method to identify dofs
  /// @param[in] check_midpoint (bool)
  DirichletBC(std::shared_ptr<const function::FunctionSpace> V,
              std::shared_ptr<const function::Function> g,
              const mesh::SubDomain& sub_domain,
              Method method = Method::topological, bool check_midpoint = true);

  /// Create boundary condition for subdomain by boundary markers (facet
  /// numbers)
  ///
  /// @param[in] V (FunctionSpace)
  ///         The function space.
  /// @param[in] g (Function)
  ///         The value.
  /// @param[in] markers (std::vector<std:size_t>&)
  ///         Subdomain markers (facet index local to process)
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
  // Build map of shared dofs in V to dofs in Vg
  static std::map<PetscInt, PetscInt>
  shared_bc_to_g(const function::FunctionSpace& V,
                 const function::FunctionSpace& Vg);

  // Compute boundary value dofs (topological approach)
  static std::set<std::array<PetscInt, 2>>
  compute_bc_dofs_topological(const function::FunctionSpace& V,
                              const function::FunctionSpace* Vg,
                              const std::vector<std::int32_t>& facets);

  // Compute boundary values dofs (geometrical approach)
  static std::set<PetscInt>
  compute_bc_dofs_geometric(const function::FunctionSpace& V,
                            const function::FunctionSpace* Vg,
                            const std::vector<std::int32_t>& facets);

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
