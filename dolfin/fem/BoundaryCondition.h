// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2007, 2008.
// Modified by Johan Hake 2009.
//
// First added:  2008-06-18
// Last changed: 2009-11-12

#ifndef __BOUNDARY_CONDITION_H
#define __BOUNDARY_CONDITION_H

#include <vector>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/types.h>

namespace dolfin
{

  template<class T> class Array;
  class GenericMatrix;
  class GenericVector;
  class FunctionSpace;

  /// Common base class for boundary conditions

  class BoundaryCondition
  {
  public:

    /// Constructor
    BoundaryCondition(const FunctionSpace& V);

    /// Constructor
    BoundaryCondition(boost::shared_ptr<const FunctionSpace> V);

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

    /// Return function space
    const FunctionSpace& function_space() const;

    /// Return shared pointer to function space
    boost::shared_ptr<const FunctionSpace> function_space_ptr() const;

  protected:

    // Friends
    friend class AdaptiveObjects;

    // Check arguments
    void check_arguments(GenericMatrix* A,
                         GenericVector* b,
                         const GenericVector* x) const;

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

      std::vector< Array<double> > array_coordinates;

    };

    // The function space (possibly a sub function space)
    boost::shared_ptr<const FunctionSpace> _function_space;

  };

}

#endif
