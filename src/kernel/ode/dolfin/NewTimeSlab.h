// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_TIME_SLAB_H
#define __NEW_TIME_SLAB_H

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>
#include <dolfin/Alloc.h>

namespace dolfin
{
  
  /// A NewTimeSlab holds the degrees of freedom for the solution
  /// of an ODE between two synchronized time levels t0 and t1.
  ///
  /// This is a new experimental version aiming at reducing the
  /// overhead of multi-adaptive time-stepping to a minimum. This
  /// class will change name and replace the class TimeSlab.

  class NewTimeSlab
  {
  public:

    /// Create time slab with given synchronized time levels (t1 may change)
    NewTimeSlab(real t0, real t1);

    /// Destructor
    ~NewTimeSlab();
    
    /// Build time slab
    void build();

    /// Return start time
    real starttime() const;
    
    /// Return end time
    real endtime() const;
    
    /// Return length of time slab
    real length() const;

    /// Output
    friend LogStream& operator<<(LogStream& stream, const NewTimeSlab& timeslab);

  private:

    // Create time slab
    void createTimeSlab();
    void createElements();

    // Create time slab data
    void createDofs();
    void createElement();
    void createStages();
    void createSubSlab();

    // Reallocate data
    void reallocDofs(uint newsize);
    void reallocElements(uint newsize);
    void reallocStages(uint newsize);
    void reallocSubSlabs(uint newsize);

    // Compute size of data
    void computeSize(uint& nj, uint& ne, uint& nl, uint& ns) const;
    
    //--- Time slab data ---

    real* jx;  // Mapping j --> value of dof j
    uint* je;  // Mapping j --> element of dof j
    uint* jl;  // Mapping j --> stage of dof j
    uint* jd;  // Mapping j --> dependencies of dof j

    uint* ei;  // Mapping e --> index of element e
    uint* es;  // Mapping e --> sub slab of element e
    real* ej;  // Mapping e --> initial dof of element e

    real* lt;  // Mapping l --> time at stage l
    real* lw;  // Mapping l --> weights at stage l

    real* sk;  // Mapping s --> time step of sub slab s
    uint* sn;  // Mapping s --> number of stages of sub slab s
    uint* sm;  // Mapping s --> method of sub slab s

    Alloc alloc_j; // Allocation data for dofs j
    Alloc alloc_e; // Allocation data for elements e
    Alloc alloc_l; // Allocation data for stages l
    Alloc alloc_s; // Allocation data for sub slabs s

    real t0; // Start time of time slab
    real t1; // End time of time slab

  };

}

#endif
