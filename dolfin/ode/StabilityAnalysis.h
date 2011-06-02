// Copyright (C) 2009 Benjamin Kehlet
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
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

    void get_JT(real* JT, const Array<real>& u, real& t);

    ODE& ode;
    ODESolution& u;
    bool write_to_file;
    uint n;

  };
}

#endif
