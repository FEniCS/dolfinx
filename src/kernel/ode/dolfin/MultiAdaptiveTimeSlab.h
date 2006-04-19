// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-27
// Last changed: 2005-11-11

#ifndef __MULTI_ADAPTIVE_TIME_SLAB_H
#define __MULTI_ADAPTIVE_TIME_SLAB_H

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>
#include <dolfin/Alloc.h>
#include <dolfin/Partition.h>
#include <dolfin/MultiAdaptivity.h>
#include <dolfin/TimeSlab.h>

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
    real build(real a, real b);

    /// Solve time slab system
    bool solve();

    /// Check if current solution can be accepted
    bool check(bool first);
    
    /// Shift time slab (prepare for next time slab)
    bool shift();

    /// Reset to initial data
    void reset();
    
    /// Prepare sample at time t
    void sample(real t);

    /// Sample solution value of given component at given time
    real usample(uint i, real t);

    /// Sample time step size for given component at given time
    real ksample(uint i, real t);

    /// Sample residual for given component at given time
    real rsample(uint i, real t);

    /// Display time slab data
    void disp() const;

    /// Friends
    friend class MultiAdaptiveFixedPointSolver;
    friend class MultiAdaptiveNewtonSolver;
    friend class MultiAdaptiveJacobian;
    friend class UpdatedMultiAdaptiveJacobian;
    friend class MultiAdaptivePreconditioner;

  private:

    // Reallocate all data
    void allocData(real a, real b);

    // Create time slab
    real createTimeSlab(real a, real b, uint offset);

    // Create time slab data
    void create_s(real t0, real t1, uint offset, uint end);
    void create_e(uint index, uint subslab, real a, real b);
    void create_j(uint index);
    void create_d(uint index, uint element, uint subslab, real a0, real b0);
   
    // Reallocation of data
    void alloc_s(uint newsize);
    void alloc_e(uint newsize);
    void alloc_j(uint newsize);
    void alloc_d(uint newsize);

    // Compute length of time slab
    real computeEndTime(real a, real b, uint offset, uint& end);

    // Compute size of data
    real computeDataSize(real a, real b, uint offset);
    
    // Compute number of dependencies to components with smaller time steps
    uint countDependencies(uint i0);

    // Compute number of dependencies to components with smaller time steps
    uint countDependencies(uint i0, real b0);

    // Check if the given time is within the given interval
    bool within(real t, real a, real b) const;

    // Check if the first given interval is within the second interval
    bool within(real a0, real b0, real a1, real b1) const;

    // Cover all elements in sub slab and return e1, with e0 <= e < e1
    uint coverSlab(int subslab, uint e0);

    // Cover all elements in next sub slab and return next sub slab
    uint coverNext(int subslab, uint element);

    // Cover given time for all components
    void coverTime(real t);

    // Compute maximum of all element residuals
    void computeMaxResiduals();

    // Evaluate right-hand side at quadrature points of given element (cG)
    void cGfeval(real* f, uint s0, uint e0, uint i0, real a0, real b0, real k0);

    // Evaluate right-hand side at quadrature points of given element (dG)
    void dGfeval(real* f, uint s0, uint e0, uint i0, real a0, real b0, real k0);

    // Choose solver
    TimeSlabSolver* chooseSolver();

    //--- Time slab data ---

    real* sa; // Mapping s --> start time t of sub slab s
    real* sb; // Mapping s --> end time t of sub slab s
        
    uint* ei; // Mapping e --> component index i of element e
    uint* es; // Mapping e --> time slab s containing element e
    uint* ee; // Mapping e --> previous element e of element e
    uint* ed; // Mapping e --> first dependency d of element e
    
    real* jx; // Mapping j --> value of dof j
    
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

    //-- Auxiliary data ---

    TimeSlabSolver* solver;     // The solver
    MultiAdaptivity adaptivity; // Adaptive time step regulation
    Partition partition;        // Time step partitioning 
    int* elast;                 // Last element for each component
    real* u;                    // Interpolated solution vector
    real* f0;                   // Right-hand side at left end-point for cG
    uint emax;                  // Last covered element for sample
    real kmin;                  // Minimum time step (exluding threshold modified)

    // FIXME: Temporary data that we should get rid of

    // FIXME: Not needed, do propagation in shift(), fix later when working
    real* f0tmp;                // Temporary storage for f0 when copying

    real* kmax;                 // Maximum time steps in the time slab
    real* rmax;                 // Maximum residuals in the time slab
    real* krmax;                // Maximum local error (k*r) in the time slab

    // Values of right-hand side at quadrature points
    real* ftmp; 

  };

}

#endif
