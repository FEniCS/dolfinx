// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_TIME_SLAB_H
#define __NEW_TIME_SLAB_H

namespace dolfin
{

  class ODE;
  class NewMethod;

  /// This is the base class for time slabs, i.e., the collection of
  /// the degrees of freedom for the solution of an ODE between to
  /// time levels a and b.

  class NewTimeSlab
  {
  public:

    /// Constructor
    NewTimeSlab(ODE& ode);

    /// Destructor
    virtual ~NewTimeSlab();
    
    /// Build time slab, return end time
    virtual real build(real a, real b) = 0;

    /// Solve time slab system
    virtual void solve() = 0;

    /// Shift time slab (prepare for next time slab)
    virtual void shift() = 0;

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
    friend LogStream& operator<<(LogStream& stream, const NewTimeSlab& timeslab);

    /// Friends
    friend class TimeSlabJacobian;
    friend class TimeSlabSolver;

  protected:
    
    uint N;  // Size of system
    real _a; // Start time of time slab
    real _b; // End time of time slab
    
    ODE& ode;                 // The ODE
    const NewMethod* method;  // Method, mcG(q) or mdG(q)  
    real* u0;                 // Initial values

  };

}

#endif
