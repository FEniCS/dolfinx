// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2007, 2008.
//
// First added:  2008-06-18
// Last changed: 2008-11-03

#ifndef __BOUNDARY_CONDITION_H
#define __BOUNDARY_CONDITION_H

#include <dolfin/common/types.h>

namespace dolfin
{

  class GenericMatrix;
  class GenericVector;
  class SubSystem;
  class FunctionSpace;

  /// Common base class for boundary conditions

  class BoundaryCondition
  {
  public:

    /// Constructor
    BoundaryCondition(const FunctionSpace& V);

    /// Constructor
    BoundaryCondition(const FunctionSpace& V, const SubSystem& sub_system);

    /// Destructor
    virtual ~BoundaryCondition();

    /// Apply boundary condition to a matrix
    virtual void apply(GenericMatrix& A) const = 0;

    /// Apply boundary condition to a vector
    virtual void apply(GenericVector& b) const = 0;

    /// Apply boundary condition to a linear system
    virtual void apply(GenericMatrix& A, GenericVector& b) const = 0;

    /// Apply boundary condition to a vector for a nonlinear problem
    virtual void apply(GenericVector& b, const GenericVector& x) const = 0;

    /// Apply boundary condition to a linear system for a nonlinear problem
    virtual void apply(GenericMatrix& A, GenericVector& b, const GenericVector& x) const = 0;

  protected:

    // Local data for application of boundary conditions
    class LocalData
    {
    public:
      
      // Constructor
      LocalData(const FunctionSpace& V);
      
      // Destructor
      ~LocalData();

      // Local dimension
      uint n;
      
      // Coefficients
      double* w;

      // Cell dofs
      uint* cell_dofs;

      // Facet dofs
      uint* facet_dofs;

      // Coordinates for dofs
      double** coordinates;

    };

    // The function space (possibly a sub function space)
    std::tr1::shared_ptr<const FunctionSpace> V;

  };

}

#endif
