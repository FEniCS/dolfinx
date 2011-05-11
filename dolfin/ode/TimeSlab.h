// Copyright (C) 2005-2009 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2005-05-02
// Last changed: 2009-08-10

#ifndef __TIME_SLAB_H
#define __TIME_SLAB_H

#include <dolfin/common/real.h>
#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>
#include <dolfin/common/Array.h>

namespace dolfin
{

  class ODE;
  class ODESolution;
  class Method;
  class uBLASVector;
  class Lagrange;

  /// This is the base class for time slabs, the collections of
  /// degrees of freedom for the solution of an ODE between two
  /// synchronized time levels a and b.

  class TimeSlab : public Variable
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
    virtual bool shift(bool end) = 0;

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

    /// Set state for ODE
    void set_state(const real* u);

    /// Get state for ODE
    void get_state(real* u);

    const Lagrange get_trial() const;
    const real* get_quadrature_weights() const;

    /// Sample solution value of given component at given time
    virtual real usample(uint i, real t) = 0;

    /// Sample time step size for given component at given time
    virtual real ksample(uint i, real t) = 0;

    /// Sample residual for given component at given time
    virtual real rsample(uint i, real t) = 0;

    /// Save to ODESolution object
    virtual void save_solution(ODESolution& u) = 0;

    /// Friends
    friend class TimeSlabJacobian;
    friend class TimeSlabSolver;
    friend class Sample;

  protected:

    // Write given solution vector to file
    static void write(Array<real>& u);

    //TODO: Clean needed here? These functions should probably not be here.

    // Copy data of given size between vectors with given offsets
    static void copy(const real* x, uint xoffset, real* y, uint yoffset, uint n);

    // Copy data of given size between vectors with given offsets
    static void copy(const uBLASVector& x, uint xoffset, real* y, uint yoffset, uint n);

    // Copy data of given size between vectors with given offsets
    static void copy(const real* x, uint xoffset, uBLASVector& y, uint yoffset, uint n);

    // Copy data of given size between vectors with given offsets
    static void copy(const uBLASVector& x, uint xoffset, uBLASVector& y, uint yoffset, uint n);

    // Copy data of given size between vectors with given offsets
    static void copy(const uBLASVector& x, uint xoffset, Array<real>& y);

    // Copy data of given size between vectors with given offsets
    static void copy(const Array<real>& x, uBLASVector& y, uint xoffset);

    // Copy data of given size between vectors with given offsets
    static void copy(const Array<real>& x, Array<real>& y);

    // Copy data of given size between vectors with given offsets
    static void copy(const real* x, uint xoffset, Array<real>& y);

    // Copy data of given size between vectors with given offsets
    static void copy(const Array<real>& x, real* y, uint yoffset);


    uint N;  // Size of system
    real _a; // Start time of time slab
    real _b; // End time of time slab

    ODE& ode;             // The ODE
    const Method* method; // Method, mcG(q) or mdG(q)
    Array<real> u0;         // Initial values (current end-time state)

    bool save_final; // True if we should save the solution at final time

  };

}

#endif
