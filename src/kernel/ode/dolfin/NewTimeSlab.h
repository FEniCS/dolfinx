// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_TIME_SLAB_H
#define __NEW_TIME_SLAB_H

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>
#include <dolfin/NewPartition.h>
#include <dolfin/Alloc.h>

namespace dolfin
{

  class ODE;
  class Method;
  class Adaptivity;
  
  /// A NewTimeSlab holds the degrees of freedom for the solution
  /// of an ODE between two synchronized time levels t0 and t1.
  ///
  /// This is a new experimental version aiming at reducing the
  /// overhead of multi-adaptive time-stepping to a minimum. This
  /// class will change name and replace the class TimeSlab.

  class NewTimeSlab
  {
  public:

    /// Constructor
    NewTimeSlab(const ODE& ode, const Method& method);

    /// Destructor
    ~NewTimeSlab();
    
    /// Build time slab, return end time
    real build(real t0, real t1, Adaptivity& adaptivity);

    /// Return start time of time slab
    real starttime() const;
    
    /// Return end time of time slab
    real endtime() const;

    /// Return length of time slab
    real length() const;

    /// Output
    friend LogStream& operator<<(LogStream& stream, const NewTimeSlab& timeslab);

  private:

    // Create time slab
    real createTimeSlab(real t0, real t1, Adaptivity& adaptivity, uint offset);

    // Create time slab data
    void createSubSlab (real t0, uint offset, uint end);
    void createElement (uint index);
    void createDofs    (uint index);
    void createDeps    ();

    // Reallocate data
    void allocData     (real t0, real t1, Adaptivity& adaptivity);
    void allocSubSlabs (uint newsize);
    void allocElements (uint newsize);
    void allocDofs     (uint newsize);
    void allocDeps     (uint newsize);

    // Compute length of time slab
    real computeEndTime(real t0, real t1, Adaptivity& adaptivity,
			uint offset, uint& end);

    // Compute size of data
    real computeDataSize(uint& ns, uint& ne, uint& nj,
			 real t0, real t1,
			 Adaptivity& adaptivity, uint offset);
    
    //--- Time slab data ---

    uint* se; // Mapping s --> first element e of sub slab s
    real* st; // Mapping s --> start time t of sub slab s
    
    uint* ej; // Mapping e --> first dof j of element e
    uint* ei; // Mapping e --> component index i of element e

    real* jx; // Mapping j --> value of dof j
    uint* jd; // Mapping j --> first dependency d of dof j

    uint* de; // Mapping d --> element e of dependency d

    //--- Auxiliary data ---

    Alloc alloc_s; // Allocation data for sub slabs s
    Alloc alloc_e; // Allocation data for elements e
    Alloc alloc_j; // Allocation data for dofs j
    Alloc alloc_d; // Allocation data for dependencies d

    const ODE& ode;         // The ODE
    const Method& method;   // Method, mcG(q) or mdG(q)
    NewPartition partition; // Time step partitioning 

    real _t0; // Start time of time slab
    real _t1; // End time of time slab

  };

}

#endif
