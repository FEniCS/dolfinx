// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-05-02
// Last changed: 2005-10-29

#ifndef __TIME_SLAB_H
#define __TIME_SLAB_H

namespace dolfin
{

  class ODE;
  class Method;

  /// This is the base class for time slabs, i.e., the collection of
  /// the degrees of freedom for the solution of an ODE between to
  /// time levels a and b.

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
    
    uint N;  // Size of system
    real _a; // Start time of time slab
    real _b; // End time of time slab
    
    ODE& ode;             // The ODE
    const Method* method; // Method, mcG(q) or mdG(q)  
    real* u0;             // Initial values

    bool save_final; // True if we should save the solution at final time

    // Write given solution to file
    void write(const real u[]);

  };

}

#endif
