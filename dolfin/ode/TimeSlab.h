// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-05-02
// Last changed: 2006-08-21

#ifndef __TIME_SLAB_H
#define __TIME_SLAB_H

#include <dolfin/common/types.h>
#include <dolfin/la/uBlasVector.h>

namespace dolfin
{

  class ODE;
  class Method;

  /// This is the base class for time slabs, the collections of
  /// degrees of freedom for the solution of an ODE between two
  /// synchronized time levels a and b.

  class TimeSlab
  {
  public:

    /// Constructor
    TimeSlab(ODE& ode);

    /// Destructor
    virtual ~TimeSlab();
    
    /// Build time slab, return end time
    virtual real build(real a, real b) = 0;

    /// Solve time slab system
    virtual bool solve() = 0;

    /// Check if current solution can be accepted
    virtual bool check(bool first) = 0;

    /// Shift time slab (prepare for next time slab)
    virtual bool shift() = 0;

    /// Prepare sample at time t
    virtual void sample(real t) = 0;

    /// Return number of components
    uint size() const;

    /// Return start time of time slab
    real starttime() const;
    
    /// Return end time of time slab
    real endtime() const;

    /// Return length of time slab
    real length() const;

    /// Sample solution value of given component at given time
    virtual real usample(uint i, real t) = 0;

    /// Sample time step size for given component at given time
    virtual real ksample(uint i, real t) = 0;

    /// Sample residual for given component at given time
    virtual real rsample(uint i, real t) = 0;

    /// Display time slab data
    virtual void disp() const = 0;

    /// Output
    friend LogStream& operator<<(LogStream& stream, const TimeSlab& timeslab);

    /// Friends
    friend class TimeSlabJacobian;
    friend class TimeSlabSolver;

  protected:

    // Write given solution to file
    static void write(const uBlasVector& u);

    // Copy data of given size between vectors with given offsets
    static void copy(const real x[], uint xoffset, real y[], uint yoffset, uint n);

    // Copy data of given size between vectors with given offsets
    static void copy(const uBlasVector& x, uint xoffset, real y[], uint yoffset, uint n);

    // Copy data of given size between vectors with given offsets
    static void copy(const real x[], uint xoffset, uBlasVector& y, uint yoffset, uint n);

    // Copy data of given size between vectors with given offsets
    static void copy(const uBlasVector& x, uint xoffset, uBlasVector& y, uint yoffset, uint n);
    
    uint N;  // Size of system
    real _a; // Start time of time slab
    real _b; // End time of time slab
    
    ODE& ode;             // The ODE
    const Method* method; // Method, mcG(q) or mdG(q)  
    uBlasVector u0;       // Initial values
    
    bool save_final; // True if we should save the solution at final time

  };

}

#endif
