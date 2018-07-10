// Copyright (C) 2007-2012 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Variable.h>
#include <dolfin/common/types.h>
#include <map>
#include <memory>
#include <unordered_map>

namespace dolfin
{

namespace function
{
class GenericFunction;
class FunctionSpace;
} // namespace function

namespace mesh
{
class Facet;
template <typename T>
class MeshFunction;
class SubDomain;
} // namespace mesh

namespace fem
{

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
/// Alternatively, the boundary may be specified by a _mesh::MeshFunction_
/// over facets labeling all mesh facets together with a number that
/// specifies which facets should be included in the boundary.
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
///
/// Note that there may be caching employed in BC computation for
/// performance reasons. In particular, applicable DOFs are cached
/// by some methods on a first apply(). This means that changing a
/// supplied object (defining boundary subdomain) after first use may
/// have no effect. But this is implementation and method specific.

class DirichletBC : public common::Variable
{

public:
  /// map type used by DirichletBC
  typedef std::unordered_map<std::size_t, PetscScalar> Map;

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
  /// @param[in] g (GenericFunction)
  ///         The value
  /// @param[in] sub_domain (mesh::SubDomain)
  ///         The subdomain
  /// @param[in] method (std::string)
  ///         Optional argument: A string specifying
  ///         the method to identify dofs
  /// @param[in] check_midpoint (bool)
  DirichletBC(std::shared_ptr<const function::FunctionSpace> V,
              std::shared_ptr<const function::GenericFunction> g,
              std::shared_ptr<const mesh::SubDomain> sub_domain,
              Method method = Method::topological, bool check_midpoint = true);

  /// Create boundary condition for subdomain specified by index
  ///
  /// @param[in] V (FunctionSpace)
  ///         The function space.
  /// @param[in] g (GenericFunction)
  ///         The value.
  /// @param[in] sub_domains (mesh::MeshFnunction<std::size_t>)
  ///         Subdomain markers
  /// @param[in] sub_domain (std::size_t)
  ///         The subdomain index (number)
  /// @param[in] method (std::string)
  ///         Optional argument: A string specifying the
  ///         method to identify dofs.
  DirichletBC(
      std::shared_ptr<const function::FunctionSpace> V,
      std::shared_ptr<const function::GenericFunction> g,
      std::shared_ptr<const mesh::MeshFunction<std::size_t>> sub_domains,
      std::size_t sub_domain, Method method = Method::topological);

  /// Create boundary condition for subdomain by boundary markers
  /// (cells, local facet numbers)
  ///
  /// @param[in] V (FunctionSpace)
  ///         The function space.
  /// @param[in] g (GenericFunction)
  ///         The value.
  /// @param[in] markers (std::vector<std:size_t>&)
  ///         Subdomain markers (facet index local to process)
  /// @param[in] method (std::string)
  ///         Optional argument: A string specifying the
  ///         method to identify dofs.
  DirichletBC(std::shared_ptr<const function::FunctionSpace> V,
              std::shared_ptr<const function::GenericFunction> g,
              const std::vector<std::size_t>& markers,
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

  /// Get Dirichlet dofs and values. If a method other than 'pointwise' is
  /// used in parallel, the map may not be complete for local vertices since
  /// a vertex can have a bc applied, but the partition might not have a
  /// facet on the boundary. To ensure all local boundary dofs are marked,
  /// it is necessary to call gather() on the returned boundary values.
  ///
  /// @param[in,out] boundary_values (Map&)
  ///         Map from dof to boundary value.
  void get_boundary_values(Map& boundary_values) const;

  /// Get boundary values from neighbour processes. If a method other than
  /// "pointwise" is used, this is necessary to ensure all boundary dofs are
  /// marked on all processes.
  ///
  /// @param[in,out] boundary_values (Map&)
  ///         Map from dof to boundary value.
  void gather(Map& boundary_values) const;

  /// Return boundary markers
  ///
  /// @return std::vector<std::size_t>&
  ///         Boundary markers (facets stored as pairs of cells and
  ///         local facet numbers).
  const std::vector<std::size_t>& markers() const;

  /// Return function space V
  ///
  /// @return FunctionSpace
  ///         The function space to which boundary conditions are applied.
  std::shared_ptr<const function::FunctionSpace> function_space() const
  {
    return _function_space;
  }

  /// Return boundary value g
  ///
  /// @return GenericFunction
  ///         The boundary values.
  std::shared_ptr<const function::GenericFunction> value() const;

  /// Return shared pointer to subdomain
  ///
  /// @return mesh::SubDomain
  ///         Shared pointer to subdomain.
  std::shared_ptr<const mesh::SubDomain> user_sub_domain() const;

  /// Set value g for boundary condition, domain remains unchanged
  ///
  /// @param[in] g (GenericFucntion)
  ///         The value.
  void set_value(std::shared_ptr<const function::GenericFunction> g);

  /// Set value to 0.0
  void homogenize();

  /// Return method used for computing Dirichlet dofs
  ///
  /// @return std::string
  ///         Method used for computing Dirichlet dofs ("topological",
  ///         "geometric" or "pointwise").
  Method method() const;

private:
  class LocalData;

  // Check input data to constructor
  void check() const;

  // Initialize facets (from sub domain, mesh, etc)
  void init_facets(const MPI_Comm mpi_comm) const;

  // Initialize sub domain markers from sub domain
  void
  init_from_sub_domain(std::shared_ptr<const mesh::SubDomain> sub_domain) const;

  // Initialize sub domain markers from mesh::MeshFunction
  void
  init_from_mesh_function(const mesh::MeshFunction<std::size_t>& sub_domains,
                          std::size_t sub_domain) const;

  // Compute boundary values for facet (topological approach)
  void compute_bc_topological(Map& boundary_values, LocalData& data) const;

  // Compute boundary values for facet (geometrical approach)
  void compute_bc_geometric(Map& boundary_values, LocalData& data) const;

  // Compute boundary values for facet (pointwise approach)
  void compute_bc_pointwise(Map& boundary_values, LocalData& data) const;

  // Check if the point is in the same plane as the given facet
  bool on_facet(const Eigen::Ref<EigenArrayXd>, const mesh::Facet& facet) const;

  // The function space (possibly a sub function space)
  std::shared_ptr<const function::FunctionSpace> _function_space;

  // The function
  std::shared_ptr<const function::GenericFunction> _g;

  // Search method
  Method _method;

  // User defined sub domain
  std::shared_ptr<const mesh::SubDomain> _user_sub_domain;

  // Cached number of bc dofs, used for memory allocation on second use
  mutable std::size_t _num_dofs;

  // Boundary facets, stored by facet index (local to process)
  mutable std::vector<std::size_t> _facets;

  // Cells attached to boundary, stored by cell index with map to
  // local dof number
  mutable std::map<std::size_t, std::vector<std::size_t>> _cells_to_localdofs;

  // User defined mesh function
  std::shared_ptr<const mesh::MeshFunction<std::size_t>> _user_mesh_function;

  // User defined sub domain marker for mesh or mesh function
  std::size_t _user_sub_domain_marker;

  // Flag for whether midpoints should be checked
  bool _check_midpoint;

  // Local data for application of boundary conditions
  class LocalData
  {
  public:
    // Constructor
    LocalData(const function::FunctionSpace& V);

    // Coefficients
    std::vector<PetscScalar> w;

    // mesh::Facet dofs
    std::vector<int> facet_dofs;

    // Coordinates for dofs
    EigenRowArrayXXd coordinates;
  };
};
} // namespace fem
} // namespace dolfin
