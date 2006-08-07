// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-27
// Last changed: 2006-07-06

#ifndef __MULTI_ADAPTIVE_JACOBIAN_H
#define __MULTI_ADAPTIVE_JACOBIAN_H

#include <dolfin/TimeSlabJacobian.h>

namespace dolfin
{
  
  class MultiAdaptiveNewtonSolver;
  class MultiAdaptiveTimeSlab;
    
  /// This class represents the Jacobian matrix of the system of
  /// equations defined on a multi-adaptive time slab.

  class MultiAdaptiveJacobian : public TimeSlabJacobian
  {
  public:

    /// Constructor
    MultiAdaptiveJacobian(MultiAdaptiveNewtonSolver& newton,
			  MultiAdaptiveTimeSlab& timeslab);

    /// Destructor
    ~MultiAdaptiveJacobian();
    
    /// Return number of rows (dim = 0) or columns (dim = 1)
    uint size(const uint dim) const;

    /// Compute product y = Ax
    void mult(const uBlasVector& x, uBlasVector& y) const;

    /// Recompute Jacobian if necessary
    void update();

    /// Friends
    friend class MultiAdaptivePreconditioner;

  private:

    // Compute product for mcG(q)
    void cGmult(const uBlasVector& x, uBlasVector& y) const;

    // Compute product for mdG(q)
    void dGmult(const uBlasVector& x, uBlasVector& y) const;

    // The Newton solver
    MultiAdaptiveNewtonSolver& newton;

    // The time slab
    MultiAdaptiveTimeSlab& ts;

    // Values of the Jacobian df/du of the right-hand side
    real* Jvalues;

    // Indices for first element of each row for the Jacobian df/du
    uint* Jindices;
    
    // Lookup table for dependencies to components with smaller time steps
    real* Jlookup;
    
  };

}

#endif
