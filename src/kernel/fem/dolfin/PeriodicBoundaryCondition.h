// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-07-08
// Last changed: 2007-07-08

#ifndef __PERIODIC_BOUNDARY_CONDITION_H
#define __PERIODIC_BOUNDARY_CONDITION_H

#include <ufc.h>

#include <dolfin/constants.h>
#include <dolfin/SubSystem.h>

namespace dolfin
{

  class Function;
  class Mesh;
  class SubDomain;
  class Form;
  class GenericMatrix;
  class GenericVector;

  /// This class specifies the interface for setting periodic boundary
  /// conditions for partial differential equations,
  ///
  ///    u(x) = u(F^{-1}(x)) on G,
  ///    u(x) = u(F(x))      on H,
  ///
  /// where F : H --> G is a map from a subdomain H to a subdomain G.
  ///
  /// A PeriodicBoundaryCondition is specified by a Mesh and a
  /// SubDomain. The given subdomain must overload both the inside()
  /// function, which specifies the points of G, and the map()
  /// function, which specifies the map from the points of H to the
  /// points of G.
  ///
  /// For mixed systems (vector-valued and mixed elements), an
  /// optional set of parameters may be used to specify for which sub
  /// system the boundary condition should be specified.
  
  class PeriodicBoundaryCondition
  {
  public:

    /// Create periodic boundary condition for sub domain
    PeriodicBoundaryCondition(Mesh& mesh, SubDomain& sub_domain);

    /// Create sub system boundary condition for sub domain
    PeriodicBoundaryCondition(Mesh& mesh, SubDomain& sub_domain,
                      const SubSystem& sub_system);

    /// Destructor
    ~PeriodicBoundaryCondition();

    /// Apply boundary condition to linear system
    void apply(GenericMatrix& A, GenericVector& b, const Form& form);

  private:

    // The mesh
    Mesh& mesh;

    // The subdomain
    SubDomain& sub_domain;

    // Sub system
    SubSystem sub_system;

  };

}

#endif
