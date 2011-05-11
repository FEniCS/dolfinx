// Copyright (C) 2005-2008 Anders Logg
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
// First added:  2005-01-28
// Last changed: 2009-09-08

#include <dolfin/log/dolfin_log.h>
#include "Alloc.h"
#include "Method.h"
#include "MonoAdaptiveTimeSlab.h"
#include "MonoAdaptiveFixedPointSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptiveFixedPointSolver::MonoAdaptiveFixedPointSolver
(MonoAdaptiveTimeSlab& timeslab)
  : TimeSlabSolver(timeslab), ts(timeslab), xold(0),
    stabilize(ode.parameters["fixed-point_stabilize"]), mi(0), li(0), ramp(1.0)
{
  rampfactor = ode.parameters["fixed-point_stabilization_ramp"].get_real();

  alpha = ode.parameters["fixed-point_damping"].get_real();

  // Initialize old values at right end-point
  xold = new real[ts.N];
  for (uint i = 0; i < ts.N; i++)
    xold[i] = 0.0;
}
//-----------------------------------------------------------------------------
MonoAdaptiveFixedPointSolver::~MonoAdaptiveFixedPointSolver()
{
  delete [] xold;
}
//-----------------------------------------------------------------------------
real MonoAdaptiveFixedPointSolver::iteration(const real& tol, uint iter,
                                             const real& d0, const real& d1)
{
  // FIXME: Cleanup stabilization

  real alpha_orig = alpha;
  if (stabilize)
  {
    if (iter == 0)
    {
      ramp = 1.0;
      mi = 0;
      li = 0;
    }

    if (iter == 0 || (d1 > d0 && li == 0))
    {
      ramp = 1.0;
      mi = ode.parameters["fixed-point_stabilization_m"];
      //mi = (int)ceil(log10(K * 1.0e4));
    }

    if (mi == 0 && li == 0)
    {
      // Choose number of ramping iterations
      li = ode.parameters["fixed-point_stabilization_l"];
    }

    if (mi == 0)
    {
      // Ramping
      ramp = ramp * rampfactor;
    }

    alpha *= ramp;
  }

  // Compute size of time step
  const real k = ts.length();

  // Save old values
  const uint xoffset = (method.nsize() - 1) * ts.N;
  ts.copy(ts.x, xoffset, xold, 0, ts.N);

  // Save norm of old solution
  xnorm = 0.0;
  for (uint j = 0; j < ts.nj; j++)
    xnorm = real_max(xnorm, real_abs(ts.x[j]));

  // Evaluate right-hand side at all quadrature points
  for (uint m = 0; m < method.qsize(); m++)
    ts.feval(m);

  // Update the values at each stage
  for (uint n = 0; n < method.nsize(); n++)
  {
    const uint noffset = n * ts.N;

    // Reset values to initial data
    for (uint i = 0; i < ts.N; i++)
      ts.x[noffset + i] += alpha*(ts.u0[i] - ts.x[noffset+i]);

    // Add weights of right-hand side
    for (uint m = 0; m < method.qsize(); m++)
    {
      const real tmp = k * method.nweight(n, m);
      const uint moffset = m * ts.N;
      for (uint i = 0; i < ts.N; i++)
	ts.x[noffset + i] += alpha*tmp*ts.fq[moffset + i];
    }
  }

  // Compute size of increment
  real max_increment = 0.0;
  for (uint i = 0; i < ts.N; i++)
  {
    const real increment = real_abs(ts.x[xoffset + i] - xold[i]);
    if (increment > max_increment)
      max_increment = increment;
  }

  if (stabilize)
  {
    alpha = alpha_orig;

    if (mi > 0)
      mi -= 1;
    if (li > 0)
      li -= 1;
  }

  return max_increment;
}
//-----------------------------------------------------------------------------
dolfin::uint MonoAdaptiveFixedPointSolver::size() const
{
  return ts.nj;
}
//-----------------------------------------------------------------------------
