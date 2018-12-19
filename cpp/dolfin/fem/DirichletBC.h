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
#include <unordered_map>
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
class GenericDofMap;

/// Interface for setting (strong) Dirichlet boundary conditions.
///
///     u = g on G,
///
/// where u is the solution to be computed, g is a function
/// and G is a sub domain of the mesh.
///
/// A DirichletBC is specified by the function g, the function space
/// (trial space) and boundary indicators on (a subset of) the mesh
/// boundary.
///
/// The boundary indicators may be specified in a number of
/// different ways.
///
/// The simplest approach is to specify the boundary by a _SubDomain_
/// object, using the inside() function to specify on which facets
/// the boundary conditions should be applied. The boundary facets
/// will then be searched for and marked *only* on the first call to
/// apply. This means that the mesh could be moved after the first
/// apply and the boundary markers would still remain intact.
///
/// The 'method' variable may be used to specify the type of method
/// used to identify degrees of freedom on the boundary. Available
/// methods are: topological approach (default), geometric approach,
/// and pointwise approach. The topological approach is faster, but
/// will only identify degrees of freedom that are located on a
/// facet that is entirely on the boundary. In particular, the
/// topological approach will not identify degrees of freedom for
/// discontinuous elements (which are all internal to the cell). A
/// remedy for this is to use the geometric approach. In the
/// geometric approach, each dof on each facet that matches the
/// boundary condition will be checked. To apply pointwise boundary
/// conditions e.g. pointloads, one will have to use the pointwise
/// approach. The three possibilities are "topological", "geometric"
/// and "pointwise".
///
/// Note: when using "pointwise", the boolean argument `on_boundary`
/// in SubDomain::inside will always be false.
///
/// The 'check_midpoint' variable can be used to decide whether or
/// not the midpoint of each facet should be checked when a
/// user-defined _SubDomain_ is used to define the domain of the
/// boundary condition. By default, midpoints are always checked.
/// Note that this variable may be of importance close to corners,
/// in which case it is sometimes important to check the midpoint to
/// avoid including facets "on the diagonal close" to a corner. This
/// variable is also of importance for curved boundaries (like on a
/// sphere or cylinder), in which case it is important *not* to
/// check the midpoint which will be located in the interior of a
/// domain defined relative to a radius.

class DirichletBC
{

public:
  /// map type used by DirichletBC
  // typedef std::unordered_map<std::size_t, PetscScalar> Map;
  typedef std::map<std::size_t, PetscScalar> Map;

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

  /// Get Dirichlet dofs and values.
  ///
  /// @param[in,out] boundary_values (Map&)
  ///         Map from dof to boundary value.
  void get_boundary_values(Map& boundary_values) const;

  /// Return function space V
  ///
  /// @return FunctionSpace
  ///         The function space to which boundary conditions are applied.
  std::shared_ptr<const function::FunctionSpace> function_space() const;

  /// Return boundary value g
  ///
  /// @return Function
  ///         The boundary values.
  std::shared_ptr<const function::Function> value() const;

  // dof indices. Must be sorted
  const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>
  dof_indices() const;

  /// Set bc entries in x to scale*x_bc
  void set(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x,
           double scale = 1.0) const;

  /// Set bc entries in x to scale*(x0 - x_bc)
  void set(
      Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x,
      const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x0,
      double scale = 1.0) const;

  // tmp
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

  // Compute boundary values for facet (pointwise approach)
  // void compute_bc_pointwise(Map& boundary_values, LocalData& data) const;

  // The function space (possibly a sub function space)
  std::shared_ptr<const function::FunctionSpace> _function_space;

  // The function
  std::shared_ptr<const function::Function> _g;

  // Dof indices in _function_space and g space to which bcs are applied, i.e.
  // u[dofs[i][0]] = g[dofs[i][1]]
  std::vector<std::array<PetscInt, 2>> _dofs;

  // Indices in _function_space to which bcs are applied
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> _dof_indices;
};
} // namespace fem
} // namespace dolfin
