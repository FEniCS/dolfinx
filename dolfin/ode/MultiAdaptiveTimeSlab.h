// Copyright (C) 2005-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-27
// Last changed: 2008-10-06

#ifndef __MULTI_ADAPTIVE_TIME_SLAB_H
#define __MULTI_ADAPTIVE_TIME_SLAB_H

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/types.h>
#include <dolfin/la/uBLASVector.h>
#include "Alloc.h"
#include "Partition.h"
#include "MultiAdaptivity.h"
#include "TimeSlab.h"

namespace dolfin
{
  class ODE;
  class Method;
  class TimeSlabSolver;
  
  /// This class represents a multi-adaptive time slab holding the
  /// degrees of freedom for the solution of an ODE between two
  /// synchronized time levels a and b, with individual time steps for
  /// the different components of the system.

  class MultiAdaptiveTimeSlab : public TimeSlab
  {
  public:

    /// Constructor
    MultiAdaptiveTimeSlab(ODE& ode);

    /// Destructor
    ~MultiAdaptiveTimeSlab();
    
    /// Build time slab, return end time
    double build(double a, double b);

    /// Solve time slab system
    bool solve();

    /// Check if current solution can be accepted
    bool check(bool first);
    
    /// Shift time slab (prepare for next time slab)
    bool shift(bool end);

    /// Reset to initial data
    void reset();
    
    /// Prepare sample at time t
    void sample(double t);

    /// Sample solution value of given component at given time
    double usample(uint i, double t);

    /// Sample time step size for given component at given time
    double ksample(uint i, double t);

    /// Sample residual for given component at given time
    double rsample(uint i, double t);

    /// Display time slab data
    void disp() const;

    /// Friends
    friend class MultiAdaptiveFixedPointSolver;
    friend class MultiAdaptiveNewtonSolver;
    friend class MultiAdaptiveJacobian;
    friend class UpdatedMultiAdaptiveJacobian;
    friend class MultiAdaptivePreconditioner;
    friend class MultiAdaptivity;

  private:

    // Reallocate all data
    void allocData(double a, double b);

    // Create time slab
    double createTimeSlab(double a, double b, uint offset);

    // Create time slab data
    void create_s(double t0, double t1, uint offset, uint end);
    void create_e(uint index, uint subslab, double a, double b);
    void create_j(uint index);
    void create_d(uint index, uint element, uint subslab, double a0, double b0);
   
    // Reallocation of data
    void alloc_s(uint newsize);
    void alloc_e(uint newsize);
    void alloc_j(uint newsize);
    void alloc_d(uint newsize);

    // Compute length of time slab
    double computeEndTime(double a, double b, uint offset, uint& end);

    // Compute size of data
    double computeDataSize(double a, double b, uint offset);
    
    // Compute number of dependencies to components with smaller time steps
    uint countDependencies(uint i0);

    // Compute number of dependencies to components with smaller time steps
    uint countDependencies(uint i0, double b0);

    // Check if the given time is within the given interval
    bool within(double t, double a, double b) const;

    // Check if the first given interval is within the second interval
    bool within(double a0, double b0, double a1, double b1) const;

    // Cover all elements in sub slab and return e1, with e0 <= e < e1
    uint coverSlab(int subslab, uint e0);

    // Cover all elements in next sub slab and return next sub slab
    uint coverNext(int subslab, uint element);

    // Cover given time for all components
    void coverTime(double t);

    // Compute maximum of all element residuals
    double computeMaxResiduals();

    // Evaluate right-hand side at quadrature points of given element (cG)
    void cGfeval(double* f, uint s0, uint e0, uint i0, double a0, double b0, double k0);

    // Evaluate right-hand side at quadrature points of given element (dG)
    void dGfeval(double* f, uint s0, uint e0, uint i0, double a0, double b0, double k0);

    // Choose solver
    TimeSlabSolver* chooseSolver();

    //--- Time slab data ---

    double* sa; // Mapping s --> start time t of sub slab s
    double* sb; // Mapping s --> end time t of sub slab s
        
    uint* ei; // Mapping e --> component index i of element e
    uint* es; // Mapping e --> time slab s containing element e
    uint* ee; // Mapping e --> previous element e of element e
    uint* ed; // Mapping e --> first dependency d of element e
    
    double* jx; // Mapping j --> value of dof j
    
    int* de;  // Mapping d --> element e of dependency d
        
    //--- Size of time slab data ---
    
    Alloc size_s; // Allocation data for sub slabs s
    Alloc size_e; // Allocation data for elements e
    Alloc size_j; // Allocation data for dofs j
    Alloc size_d; // Allocation data for dependencies d

    uint ns; // Number of sub slabs
    uint ne; // Number of elements
    uint nj; // Number of dofs
    uint nd; // Number of dependencies

    //--- Auxiliary data, size N ---

    TimeSlabSolver* solver;     // The solver (size N if diagonally damped)
    MultiAdaptivity adaptivity; // Adaptive time step regulation (size 3N)
    Partition partition;        // Time step partitioning (size N)
    int* elast;                 // Last element for each component (size N)
    double* f0;                 // Right-hand side at left end-point for cG (size N)
    double* u;                  // The interpolated solution at a given time

    //--- Auxiliary data ---
    uint emax;                  // Last covered element for sample
    double kmin;                  // Minimum time step (exluding threshold modified)

  };

}

#endif
