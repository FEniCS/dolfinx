// Copyright (C) 2014-2016 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 4 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2014-05-12
// Last changed: 2016-03-02

#ifndef __MULTI_MESH_DIRICHLET_BC_H
#define __MULTI_MESH_DIRICHLET_BC_H

#include <vector>
#include <memory>
#include <dolfin/mesh/SubDomain.h>

namespace dolfin
{

  // Forward declarations
  class MultiMeshFunctionSpace;
  class GenericFunction;
  class SubDomain;
  class GenericMatrix;
  class GenericVector;
  class DirichletBC;

  /// This class is used to set Dirichlet boundary conditions for
  /// multimesh function spaces.

  class MultiMeshDirichletBC
  {
  public:

    /// Create boundary condition for subdomain
    ///
    /// @param    V (_MultiMeshFunctionSpace_)
    ///         The function space
    /// @param    g (_GenericFunction_)
    ///         The value
    /// @param     sub_domain (_SubDomain_)
    ///         The subdomain
    /// @param     method (std::string)
    ///         Option passed to DirichletBC.
    /// @param     check_midpoint (bool)
    ///         Option passed to DirichletBC.
    /// @param     exclude_overlapped_boundaries (bool)
    ///         If true, then the variable on_boundary will
    ///         be set to false for facets that are overlapped
    ///         by another mesh (irrespective of the layering order
    ///         of the meshes).
    MultiMeshDirichletBC(std::shared_ptr<const MultiMeshFunctionSpace> V,
                         std::shared_ptr<const GenericFunction> g,
                         std::shared_ptr<const SubDomain> sub_domain,
                         std::string method="topological",
                         bool check_midpoint=true,
                         bool exclude_overlapped_boundaries=true);

    /// Create boundary condition for subdomain specified by index
    ///
    /// @param     V (_FunctionSpace_)
    ///         The function space.
    /// @param     g (_GenericFunction_)
    ///         The value.
    /// @param     sub_domains (_MeshFunction_ <std::size_t>)
    ///         Subdomain markers
    /// @param     sub_domain (std::size_t)
    ///         The subdomain index (number)
    /// @param     part (std::size_t)
    ///         The part on which to set boundary conditions
    /// @param     method (std::string)
    ///         Optional argument: A string specifying the
    ///         method to identify dofs.
    MultiMeshDirichletBC(std::shared_ptr<const MultiMeshFunctionSpace> V,
                         std::shared_ptr<const GenericFunction> g,
                         std::shared_ptr<const MeshFunction<std::size_t>> sub_domains,
                         std::size_t sub_domain,
                         std::size_t part,
                         std::string method="topological");

    /// Destructor
    ~MultiMeshDirichletBC();

    /// Apply boundary condition to a matrix
    ///
    /// @param     A (_GenericMatrix_)
    ///         The matrix to apply boundary condition to.
    void apply(GenericMatrix& A) const;

    /// Apply boundary condition to a vector
    ///
    /// @param     b (_GenericVector_)
    ///         The vector to apply boundary condition to.
    void apply(GenericVector& b) const;

    /// Apply boundary condition to a linear system
    ///
    /// @param     A (_GenericMatrix_)
    ///         The matrix to apply boundary condition to.
    /// @param     b (_GenericVector_)
    ///         The vector to apply boundary condition to.
    void apply(GenericMatrix& A,
               GenericVector& b) const;

    /// Apply boundary condition to vectors for a nonlinear problem
    ///
    /// @param    b (_GenericVector_)
    ///         The vector to apply boundary conditions to.
    /// @param     x (_GenericVector_)
    ///         Another vector (nonlinear problem).
    void apply(GenericVector& b,
               const GenericVector& x) const;

    /// Apply boundary condition to a linear system for a nonlinear problem
    ///
    /// @param     A (_GenericMatrix_)
    ///         The matrix to apply boundary conditions to.
    /// @param     b (_GenericVector_)
    ///         The vector to apply boundary conditions to.
    /// @param     x (_GenericVector_)
    ///         Another vector (nonlinear problem).
    void apply(GenericMatrix& A,
               GenericVector& b,
               const GenericVector& x) const;

  private:

    // Subclass of SubDomain wrapping user-defined subdomain
    class MultiMeshSubDomain : public SubDomain
    {
    public:

      // Constructor
      MultiMeshSubDomain(std::shared_ptr<const SubDomain> sub_domain,
                         std::shared_ptr<const MultiMesh> multimesh,
                         bool exclude_overlapped_boundaries);

      // Destructor
      ~MultiMeshSubDomain();

      // Callback for checking whether point is in domain
      bool inside(const Array<double>& x, bool on_boundary) const;

      // Set current part
      void set_current_part(std::size_t current_part);

    private:

      // User-defined subdomain
      std::shared_ptr<const SubDomain> _user_sub_domain;

      // Multimesh
      std::shared_ptr<const MultiMesh> _multimesh;

      // Current part
      std::size_t _current_part;

      // Check whether to exclude boundaries overlapped by other meshes
      bool _exclude_overlapped_boundaries;

    };

    // List of boundary conditions for parts
    std::vector<std::shared_ptr<const DirichletBC>> _bcs;

    // Wrapper of user-defined subdomain
    mutable std::shared_ptr<MultiMeshSubDomain> _sub_domain;

    // Check whether to exclude boundaries overlapped by other meshes
    bool _exclude_overlapped_boundaries;

  };

}

#endif
