// Copyright (C) 2007 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-10
// Last changed: 2007-04-26

#ifndef __NEW_BOUNDARY_CONDITION_H
#define __NEW_BOUNDARY_CONDITION_H

#include <ufc.h>

#include <dolfin/constants.h>
#include <dolfin/SubSystem.h>
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
  ///
  /// For mixed systems (vector-valued and mixed elements), an
  /// optional set of parameters may be used to specify for which sub
  /// system the boundary condition should be specified.
  
  class BoundaryCondition
  {
  public:

    /// Create boundary condition for sub domain
    BoundaryCondition(Function& g,
                      Mesh& mesh,
                      SubDomain& sub_domain);

    /// Create boundary condition for sub domain specified by index
    BoundaryCondition(Function& g,
                      MeshFunction<uint>& sub_domains, uint sub_domain);

    /// Create sub system boundary condition for sub domain
    BoundaryCondition(Function& g,
                      Mesh& mesh,
                      SubDomain& sub_domain,
                      const SubSystem& sub_system);

    /// Create sub system boundary condition for sub domain specified by index
    BoundaryCondition(Function& g,
                      MeshFunction<uint>& sub_domains,
                      uint sub_domain,
                      const SubSystem& sub_system);

    /// Destructor
    ~BoundaryCondition();

    /// Apply boundary condition to linear system
    void apply(GenericMatrix& A, GenericVector& b, const Form& form);

  private:

    // Initialize sub domain markers    
    void init(SubDomain& sub_domain);

    /*
    // Local data for application of boundary conditions
    class LocalData
    {
    public:
      
      // Constructor
      LocalData(const ufc::form& form);
      
      // Destructor
      ~LocalData();

    };
    */

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

    // Sub system
    SubSystem sub_system;

  };

}

#endif
