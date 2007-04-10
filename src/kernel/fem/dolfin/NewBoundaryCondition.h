// Copyright (C) 2007 Anders Logg and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-04-10
// Last changed: 2007-04-10

#ifndef __NEW_BOUNDARY_CONDITION_H
#define __NEW_BOUNDARY_CONDITION_H

#include <ufc.h>

#include <dolfin/constants.h>
#include <dolfin/MeshFunction.h>

namespace dolfin
{

  class Function;
  class Mesh;
  class SubDomain;
  class Form;
  class GenericMatrix;
  class GenericVector;

  /// This class specifies the interface for setting (strong)
  /// Dirichlet boundary conditions for partial differential
  /// equations,
  ///
  ///    u = g on G,
  ///
  /// where u is the solution to be computed, g is a function
  /// and G is a sub domain of the mesh.
  ///
  /// A BoundaryCondition is specified by a Function, a Mesh,
  /// a MeshFunction<uint> over the facets of the mesh and
  /// an integer sub_domain specifying the sub domain on which
  /// the boundary condition is to be applied.
  
  class NewBoundaryCondition
  {
  public:

    /// Create boundary condition from sub domain
    NewBoundaryCondition(Function& g, Mesh& mesh, SubDomain& sub_domain);

    /// Create boundary condition from sub domain markers and sub domain index
    NewBoundaryCondition(Function& g, Mesh& mesh, MeshFunction<uint>& sub_domains, uint sub_domain);

    /// Destructor
    ~NewBoundaryCondition();

    /// Apply boundary condition to linear system
    void apply(GenericMatrix& A, GenericVector& b, const Form& form);

    /// Apply boundary condition to linear system
    void apply(GenericMatrix& A, GenericVector& b, const ufc::form& form);

  private:

    // The function
    Function& g;

    // The mesh
    Mesh& mesh;

    // Sub domain markers (if any)
    MeshFunction<uint>* sub_domains;

    // The sub domain
    uint sub_domain;

    // True if sub domain markers are created locally
    bool sub_domains_local;

  };

}

#endif
