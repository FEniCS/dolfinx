// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-28
// Last changed: 2006-08-08

#ifndef __TIME_SLAB_JACOBIAN_H
#define __TIME_SLAB_JACOBIAN_H

#include <dolfin/dolfin_log.h>
#include <dolfin/uBlasDenseMatrix.h>
#include <dolfin/uBlasVector.h>
#include <dolfin/uBlasKrylovMatrix.h>

namespace dolfin
{
  
  class ODE;
  class Method;
  class TimeSlab;
    
  /// This is the base class for Jacobians defined on mono- or
  /// multi-adaptive time slabs.

  class TimeSlabJacobian : public uBlasKrylovMatrix
  {
  public:

    /// Constructor
    TimeSlabJacobian(TimeSlab& timeslab);

    /// Destructor
    ~TimeSlabJacobian();
    
    /// Return number of rows (dim = 0) or columns (dim = 1)
    virtual uint size(const uint dim) const = 0;
    
    /// Compute product y = Ax
    virtual void mult(const uBlasVector& x, uBlasVector& y) const = 0;

    /// (Re-)initialize computation of Jacobian
    virtual void init();

    /// Update dense copy of Jacobian
    void update();

    /// Return dense copy of Jacobian
    inline const uBlasDenseMatrix& matrix() const
    {
      dolfin_assert(A.size(0) == ode.size());
      return A;
    }
    
  protected:
    
    // The ODE
    ODE& ode;

    // Method, mcG(q) or mdG(q)
    const Method& method;

    // Dense copy of the Jacobian
    uBlasDenseMatrix A;

    // Vectors used to compute dense copy of the Jacobian
    uBlasVector ej, Aj;

  };

}

#endif
