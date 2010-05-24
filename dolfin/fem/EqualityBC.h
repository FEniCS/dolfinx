// Copyright (C) 2009 Bartosz Sawicki.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Hake, 2009
//
// First added:  2009-04-03
// Last changed: 2009-08-14

#ifndef __EQUALITY_BC_H
#define __EQUALITY_BC_H

#include <boost/shared_ptr.hpp>
#include <dolfin/common/types.h>
#include "BoundaryCondition.h"

namespace dolfin
{

  class Mesh;
  class SubDomain;
  class Form;
  class GenericMatrix;
  class GenericVector;

  /// This class specifies the interface for setting equality boundary
  /// conditions for partial differential equations,
  ///
  ///    u(x) = u(y),    for all x and y on G,
  ///
  /// where G is subdomain of the mesh.
  ///
  /// The sub domain G may be specified in two different ways. Both of
  /// them produce a set of unknowns (dofs) with should be equal.
  ///
  /// The simplest approach is to specify a SubDomain object, using
  /// the inside() function to specify on which facets the boundary
  /// condition should be applied.
  ///
  /// Alternatively, the boundary may be specified by the boundary
  /// indicators included in the mesh.
  ///
  /// Current implementation assume that the problem is scalar,
  /// so in case of mixed systems (vector-valued and mixed elements)
  /// all compoments will be set equal.

  class EqualityBC : public BoundaryCondition
  {
  public:

    // Create equality boundary condition for sub domain
    EqualityBC(const FunctionSpace& V,
               const SubDomain& sub_domain);

    // Create equality boundary condition for sub domain
    EqualityBC(boost::shared_ptr<const FunctionSpace> V,
               const SubDomain& sub_domain);

    // Create boundary condition for boundary data included in the mesh
    EqualityBC(const FunctionSpace& V,
               uint sub_domain);

    // Create boundary condition for boundary data included in the mesh
    EqualityBC(boost::shared_ptr<const FunctionSpace> V,
               uint sub_domain);

    // Destructor
    ~EqualityBC();

    // Apply boundary condition to a matrix
    void apply(GenericMatrix& A) const;

    // Apply boundary condition to a vector
    void apply(GenericVector& b) const;

    // Apply boundary condition to a linear system
    void apply(GenericMatrix& A, GenericVector& b) const;

    // Apply boundary condition to a vector for a nonlinear problem
    void apply(GenericVector& b, const GenericVector& x) const;

    // Apply boundary condition to a linear system for a nonlinear problem
    void apply(GenericMatrix& A, GenericVector& b, const GenericVector& x) const;

    // Initialize sub domain markers from sub domain
    void init_from_sub_domain(const SubDomain& sub_domain);

    // Initialize sub domain markers from mesh
    void init_from_mesh(uint sub_domain);

  private:

    // Vector of equal dofs
    std::vector< uint > equal_dofs;

  };

}

#endif
