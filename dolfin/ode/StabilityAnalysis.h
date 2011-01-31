// Copyright (C) 2009 Benjamin Kehlet
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-07-20
// Last changed: 2009-09-04

#ifndef __STABILITYFACTORS_H
#define __STABILITYFACTORS_H

#include <dolfin/common/types.h>
#include <dolfin/common/real.h>
#include <dolfin/common/Array.h>

namespace dolfin
{

  /// This class computes the stabilityfactors as a function of time
  /// S(t). As the stabilityfactors are defined it should solve the dual
  /// for each timestep. However, we can take advantage of the dual
  /// being linear.

  class ODESolution;
  class ODE;

  class StabilityAnalysis
  {
  public:

    /// Constructor
    StabilityAnalysis(ODE& ode, ODESolution& u);

    /// Destructor
    ~StabilityAnalysis();

    /// Compute the integral of the q'th derivative of the dual as function of (primal) endtime T
    void analyze_integral(uint q);

    /// Compute z(0) (the endpoint of the dual) as function of (primal) endtime T
    void analyze_endpoint();

  private:

    void get_JT(real* JT, const RealArray& u, real& t);

    ODE& ode;
    ODESolution& u;
    bool write_to_file;
    uint n;

  };
}

#endif
