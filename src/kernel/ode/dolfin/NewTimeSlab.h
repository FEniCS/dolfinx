// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_TIME_SLAB_H
#define __NEW_TIME_SLAB_H

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>
#include <dolfin/NewArray.h>
#include <dolfin/NewPartition.h>
#include <dolfin/NewAdaptivity.h>
#include <dolfin/Alloc.h>

namespace dolfin
{

  class ODE;
  class NewMethod;
  class TimeSlabSolver;
  
  /// A NewTimeSlab holds the degrees of freedom for the solution
  /// of an ODE between two synchronized time levels a and b.
  ///
  /// This is a new experimental version aiming at reducing the
  /// overhead of multi-adaptive time-stepping to a minimum. This
  /// class will change name and replace the class TimeSlab.

  class NewTimeSlab
  {
  public:

    /// Constructor
    NewTimeSlab(ODE& ode);

    /// Destructor
    ~NewTimeSlab();
    
    /// Build time slab, return end time
    real build(real a, real b);

    /// Solve time slab system
    void solve();

    /// Shift time slab (prepare for next time slab)
    void shift();

    /// Prepare sample at time t
    void sample(real t);

    /// Return number of components
    uint size() const;

    /// Return start time of time slab
    real starttime() const;
    
    /// Return end time of time slab
    real endtime() const;

    /// Return length of time slab
    real length() const;

    /// Sample solution value of given component at given time
    real usample(uint i, real t);

    /// Sample time step size for given component at given time
    real ksample(uint i, real t);

    /// Sample residual for given component at given time
    real rsample(uint i, real t);

    /// Display time slab data
    void disp() const;

    /// Output
    friend LogStream& operator<<(LogStream& stream, const NewTimeSlab& timeslab);

    /// Friends
    friend class FixedPointSolver;

  private:

    // Reallocate all data
    void allocData(real a, real b);

    // Create time slab
    real createTimeSlab(real a, real b, uint offset);

    // Create time slab data
    void create_s(real t0, real t1, uint offset, uint end);
    void create_e(uint index, uint subslab, real a, real b);
    void create_j(uint index);
    void create_d(uint index, uint element, real a0, real b0);
   
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

    // Cover all elements in current sub slab
    uint cover(int subslab, uint element);

    // Cover given time for all components
    void cover(real t);

    // Evaluate right-hand side at quadrature points of given element
    void feval(real* f, uint s0, uint e0, uint i0, real a0, real b0, real k0);

    //--- Time slab data ---

    // FIXME: Remove ej (not needed)

    real* sa; // Mapping s --> start time t of sub slab s
    real* sb; // Mapping s --> end time t of sub slab s
        
    uint* ej; // Mapping e --> first dof j of element e
    uint* ei; // Mapping e --> component index i of element e
    uint* es; // Mapping e --> time slab s containing element e
    uint* ee; // Mapping e --> previous element e of element e
    uint* ed; // Mapping e --> first dependency d of element e

    real* jx; // Mapping j --> value of dof j
    
    int* de;  // Mapping d --> element e of dependency d

    //--- Auxiliary data ---

    Alloc size_s; // Allocation data for sub slabs s
    Alloc size_e; // Allocation data for elements e
    Alloc size_j; // Allocation data for dofs j
    Alloc size_d; // Allocation data for dependencies d

    uint ns; // Number of sub slabs
    uint ne; // Number of elements
    uint nj; // Number of dofs
    uint nd; // Number of dependencies

    uint N;  // Size of system
    real _a; // Start time of time slab
    real _b; // End time of time slab
    
    ODE& ode;                 // The ODE
    const NewMethod* method;  // Method, mcG(q) or mdG(q)
    TimeSlabSolver* solver;   // The solver
    NewAdaptivity adaptivity; // Adaptive time step selection
    NewPartition partition;   // Time step partitioning 
    NewArray<int> elast;      // Last element for each component
    NewArray<real> u0;        // Initial values
    real* u;                  // Interpolated solution vector
    uint emax;                // Last covered element for sample

  };

}

#endif
