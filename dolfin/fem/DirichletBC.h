// Copyright (C) 2007 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Kristian Oelgaard, 2007
//
// First added:  2007-04-10
// Last changed: 2008-01-02

#ifndef __DIRICHLET_BC_H
#define __DIRICHLET_BC_H

#include <dolfin/main/constants.h>
#include "SubSystem.h"
#include <dolfin/mesh/MeshFunction.h>
#include "BoundaryCondition.h"
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/SubDomain.h>

namespace dolfin
{

  class DofMap;
  class Function;
  class Mesh;
  class Form;
  class GenericMatrix;
  class GenericVector;
  
  /// The BCMethod variable may be used to specify the type of method
  /// used to identify degrees of freedom on the boundary. Available
  /// methods are: topological approach (default), geometrical approach,
  /// and pointwise approach. The topological approach is faster,
  /// but will only identify degrees of freedom that are located on a
  /// facet that is entirely on the boundary. In particular, the
  /// topological approach will not identify degrees of freedom
  /// for discontinuous elements (which are all internal to the cell).
  /// A remedy for this is to use the geometrical approach. To apply
  /// pointwise boundary conditions e.g. pointloads, one will have to
  /// use the pointwise approach which in turn is the slowest of the
  /// three possible methods.
  enum BCMethod {topological, geometrical, pointwise};
  
  /// This class specifies the interface for setting (strong)
  /// Dirichlet boundary conditions for partial differential
  /// equations,
  ///
  ///    u = g on G,
  ///
  /// where u is the solution to be computed, g is a function
  /// and G is a sub domain of the mesh.
  ///
  /// A DirichletBC is specified by a Function, a Mesh,
  /// a MeshFunction<uint> over the facets of the mesh and
  /// an integer sub_domain specifying the sub domain on which
  /// the boundary condition is to be applied.
  ///
  /// For mixed systems (vector-valued and mixed elements), an
  /// optional set of parameters may be used to specify for which sub
  /// system the boundary condition should be specified.
  
  class DirichletBC : public BoundaryCondition
  {
  public:

    /// Create boundary condition for sub domain
    DirichletBC(Function& g,
                Mesh& mesh,
                SubDomain& sub_domain,
                BCMethod method = topological);

    /// Create boundary condition for sub domain specified by index
    DirichletBC(Function& g,
                MeshFunction<uint>& sub_domains, uint sub_domain,
                BCMethod method = topological);

    /// Create sub system boundary condition for sub domain
    DirichletBC(Function& g,
                Mesh& mesh,
                SubDomain& sub_domain,
                const SubSystem& sub_system,
                BCMethod method = topological);

    /// Create sub system boundary condition for sub domain specified by index
    DirichletBC(Function& g,
                MeshFunction<uint>& sub_domains,
                uint sub_domain,
                const SubSystem& sub_system,
                BCMethod method = topological);

    /// Simple creation of boundary condition with given value on the entire boundary
    DirichletBC(Function& g,
                Mesh& mesh,
                BCMethod method = topological);

    /// Destructor
    ~DirichletBC();

    /// Apply boundary condition to linear system
    void apply(GenericMatrix& A, GenericVector& b, const Form& form);

    /// Apply boundary condition to linear system
    void apply(GenericMatrix& A, GenericVector& b, const DofMap& dof_map, const ufc::form& form);

    /// Apply boundary condition to linear system for a nonlinear problem
    void apply(GenericMatrix& A, GenericVector& b, const GenericVector& x, const Form& form);

    /// Apply boundary condition to linear system for a nonlinear problem
    void apply(GenericMatrix& A, GenericVector& b, const GenericVector& x, const DofMap& dof_map, const ufc::form& form);

    /// Apply boundary condition to linear system for a nonlinear problem
    void zero(GenericMatrix& A, const DofMap& dof_map, const ufc::form& form);

    /// Return mesh
    Mesh& mesh();

  private:

    // Initialize sub domain markers    
    void init(SubDomain& sub_domain);

    /// Apply boundary conditions
    void apply(GenericMatrix& A, GenericVector& b,
               const GenericVector* x, const DofMap& dof_map, const ufc::form& form);

    // The function
    Function& g;

    // The mesh
    Mesh& _mesh;

    // Sub domain markers (if any)
    MeshFunction<uint>* sub_domains;

    // The sub domain
    uint sub_domain;

    // True if sub domain markers are created locally
    bool sub_domains_local;

    // Sub system
    SubSystem sub_system;

    // Search method
    BCMethod method;

    // User defined sub domain
    SubDomain* user_sub_domain;

    // Compute boundary values for facet (topological approach)
    void computeBCTopological(std::map<uint, real>& boundary_values,
                              Facet& facet,
                              BoundaryCondition::LocalData& data);
    
    // Compute boundary values for facet (geometrical approach)
    void computeBCGeometrical(std::map<uint, real>& boundary_values,
                              Facet& facet,
                              BoundaryCondition::LocalData& data);

    // Compute boundary values for facet (pointwise approach)
    void computeBCPointwise(std::map<uint, real>& boundary_values,
                              Cell& cell,
                              BoundaryCondition::LocalData& data);
    
    // Check if the point is in the same plane as the given facet
    static bool onFacet(real* coordinates, Facet& facet);
    
  };

}

#endif
