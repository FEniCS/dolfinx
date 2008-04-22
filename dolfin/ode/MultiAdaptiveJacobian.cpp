// Copyright (C) 2005-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-27
// Last changed: 2008-04-22

#include <dolfin/common/constants.h>
#include <dolfin/math/dolfin_math.h>
#include <dolfin/la/uBlasVector.h>
#include "ODE.h"
#include "Method.h"
#include "MultiAdaptiveTimeSlab.h"
#include "MultiAdaptiveNewtonSolver.h"
#include "MultiAdaptiveJacobian.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiAdaptiveJacobian::MultiAdaptiveJacobian(MultiAdaptiveNewtonSolver& newton,
					     MultiAdaptiveTimeSlab& timeslab)
  : TimeSlabJacobian(timeslab), newton(newton), ts(timeslab),
    Jvalues(0), Jindices(0), Jlookup(0)
{
  // Allocate Jacobian row indices
  Jindices = new uint[ode.size()];
  
  // Compute start of each row
  uint sum = 0;
  for (uint i = 0; i < ode.size(); i++)
  {
    Jindices[i] = sum;
    sum += ode.dependencies[i].size();
  }

  // Allocate Jacobian values
  Jvalues = new real[sum];
  for (uint pos = 0; pos < sum; pos++)
    Jvalues[pos] = 0.0;

  message("Generated Jacobian data structure for %d dependencies.", sum);

  // Compute maximum number of dependencies
  uint maxsize = 0;
  for (uint i = 0; i < ode.size(); i++)
  {
    const uint size = ode.dependencies[i].size();
    if ( size > maxsize )
      maxsize = size;
  }

  // Allocate lookup table for dependencies to components with small time steps
  Jlookup = new real[std::max(static_cast<unsigned int>(1), maxsize - 1)];
}
//-----------------------------------------------------------------------------
MultiAdaptiveJacobian::~MultiAdaptiveJacobian()
{
  if ( Jvalues ) delete [] Jvalues;
  if ( Jindices ) delete [] Jindices;
  if ( Jlookup ) delete [] Jlookup;
}
//-----------------------------------------------------------------------------
dolfin::uint MultiAdaptiveJacobian::size(uint dim) const
{
  return ts.nj;
}
//-----------------------------------------------------------------------------
void MultiAdaptiveJacobian::mult(const uBlasVector& x, uBlasVector& y) const
{
  // We iterate over all degrees of freedom j in the time slab and compute
  // y_j = (Ax)_j for each degree of freedom of the system.

  // Start with y = x, accounting for the derivative dF_j/dx_j = 1
  y = x;

  // Choose method
  if ( method.type() == Method::cG )
    cGmult(x, y);
  else
    dGmult(x, y);
}
//-----------------------------------------------------------------------------
void MultiAdaptiveJacobian::init()
{
  // Compute Jacobian at the beginning of the slab
  real t = ts.starttime();
  //message("Recomputing Jacobian matrix at t = %f.", t);
  
  // Compute Jacobian
  for (uint i = 0; i < ode.size(); i++)
  {
    const Array<uint>& deps = ode.dependencies[i];
    for (uint pos = 0; pos < deps.size(); pos++)
      Jvalues[Jindices[i] + pos] = ode.dfdu(ts.u0, t, i, deps[pos]);
  }

  /*
  // Compute Jacobian at the end of the slab
  real t = ts.endtime();
  //message("Recomputing Jacobian matrix at t = %f.", t);
  
  // Compute Jacobian
  for (uint i = 0; i < ode.size(); i++)
  {
    const Array<uint>& deps = ode.dependencies[i];
    for (uint pos = 0; pos < deps.size(); pos++)
      Jvalues[Jindices[i] + pos] = ode.dfdu(ts.u, t, i, deps[pos]);
  }
  */
}
//-----------------------------------------------------------------------------
void MultiAdaptiveJacobian::cGmult(const uBlasVector& x, uBlasVector& y) const
{
  // Reset current sub slab
  int s0 = -1;

  // Reset elast
  for (uint i = 0; i < ts.N; i++)
    ts.elast[i] = -1;

  // Iterate over all elements
  for (uint e0 = 0; e0 < ts.ne; e0++)
  {
    // Cover all elements in current sub slab
    s0 = ts.coverNext(s0, e0);
    
    // Get element data
    const uint i0 = ts.ei[e0];
    const real a0 = ts.sa[s0];
    const real b0 = ts.sb[s0];
    const real k0 = b0 - a0;
    const uint j0 = e0 * method.nsize();
    
    // Add dependency on predecessor for all dofs of element
    const int ep = ts.ee[e0];
    if ( ep != -1 )
    {
      const real xp = x(ep*method.nsize() + method.nsize() - 1);
      for (uint n = 0; n < method.nsize(); n++)
	y(j0 + n) -= xp;
    }

    // Reset Jpos
    uint Jpos = 0;

    // Iterate over dependencies for the current component
    const Array<uint>& deps = ode.dependencies[i0];
    for (uint pos = 0; pos < deps.size(); pos++)
    {
      // Get derivative
      const real dfdu = Jvalues[Jindices[i0] + pos];

      // Skip elements which have not been covered
      const uint i1 = deps[pos];
      const int e1 = ts.elast[i1];      
      if ( e1 == -1 )
      {
	// Save dependency for later
	Jlookup[Jpos++] = dfdu;
	continue;
      }

      // Skip elements with smaller time steps
      const uint s1 = ts.es[e1];
      const real b1 = ts.sb[s1];
      if ( b1 < (a0 + DOLFIN_EPS) )
      {
	// Save dependency for later
	Jlookup[Jpos++] = dfdu;

	// Add dependency to initial value for all dofs
	const real tmp = k0 * dfdu * x(e1 * method.nsize() + method.nsize() - 1);
	for (uint n = 0; n < method.nsize(); n++)
	  y(j0 + n) -= tmp * method.nweight(n, 0);

       	continue;
      }
      
      // Get first dof for other element
      const uint j1 = e1 * method.nsize();
      
      // Use fast evaluation for elements in the same sub slab
      if ( s0 == static_cast<int>(s1) )
      {
	// Add dependency to dof of initial value if any
	const int ep = ts.ee[e1];
	const real tmp0 = k0 * dfdu;
	if ( ep != -1 )
	{
	  const real tmp1 = tmp0 * x(ep * method.nsize() + method.nsize() - 1);
	  for (uint n = 0; n < method.nsize(); n++)
	    y(j0 + n) -= tmp1 * method.nweight(n, 0);
	}
	
	// Add dependencies to internal dofs
	for (uint n = 0; n < method.nsize(); n++)
	{
	  real sum = 0.0;
	  for (uint m = 0; m < method.nsize(); m++)
	    sum += method.nweight(n, m + 1) * x(j1 + m);
	  y(j0 + n) -= tmp0 * sum;
	}
      }
      else
      {
	const real a1 = ts.sa[s1];
	const real k1 = b1 - a1;
	
	// Iterate over dofs of element
	const real tmp0 = k0 * dfdu;
	for (uint n = 0; n < method.nsize(); n++)
	{
	  // Iterate over quadrature points
	  real sum = 0.0;
	  for (uint m = 0; m < method.qsize(); m++)
	  {
	    const real tau = (a0 + k0*method.qpoint(m) - a1) / k1;
	    const real tmp1 = method.nweight(n, m);
	    dolfin_assert(tau >= -DOLFIN_EPS);
	    dolfin_assert(tau <= 1.0 + DOLFIN_EPS);
	    
	    // Add dependency to dof of initial value if any
	    const int ep = ts.ee[e1];
	    if ( ep != -1 )
	    {
	      const real x0 = x(ep * method.nsize() + method.nsize() - 1);
	      sum += tmp1 * method.eval(0, tau) * x0;
	    }
	    
	    // Iterate over dofs of other element and add dependencies
	    for (uint l = 0; l < method.nsize(); l++)
	      sum += tmp1 * method.eval(l + 1, tau) * x(j1 + l);
	  }
	    
	  // Add dependencies
	  y(j0 + n) -= tmp0 * sum;
	}
      }      
    }
    
    // At this point, we have added dependencies to the predecessor,
    // to dofs in the same sub slab and to components with larger time
    // steps. It remains to add dependencies to components with
    // smaller time steps. We need to do this by iterating over
    // quadrature points, since this is the order in which the
    // dependencies are stored.

    // Get first dependency to components with smaller time steps for element
    uint d = ts.ed[e0];
    
    // Compute number of such dependencies for each nodal point
    const uint end = ( e0 < (ts.ne - 1) ? ts.ed[e0 + 1] : ts.nd );
    const uint ndep = (end - d) / method.nsize();
    dolfin_assert(ndep * method.nsize() == (end - d));

    // Iterate over quadrature points of current element
    for (uint m = 1; m < method.qsize(); m++)
    {
      // Compute quadrature point
      const real t = a0 + k0*method.qpoint(m);

      // Iterate over dependencies
      for (uint dep = 0; dep < ndep; dep++)
      {
	// Get element data
	const uint e1 = ts.de[d++];
	const uint j1 = e1 * method.nsize();
	const uint s1 = ts.es[e1];
	//const uint i1 = ts.ei[e1];
	const real a1 = ts.sa[s1];
	const real b1 = ts.sb[s1];
	const real k1 = b1 - a1;
	const real tau = (t - a1) / k1;
	
	// We don't know how to index Jvalues here and want to avoid
	// searching, but we were clever enough to pick out the value
	// before when we had the chance... :-)
	const real dfdu = Jlookup[dep];
	//message("Looks like df_%d/du_%d = %f", i0, i1, dfdu);      
	
	// Iterate over quadrature points of other element
	const real tmp0 = k0 * dfdu;
	for (uint l = 0; l < method.qsize(); l++)
	{
	  real tmp1 = tmp0 * method.eval(l, tau);
	  if ( l == 0 )
	  {
	    const int ep = ts.ee[e1];
	    if ( ep != -1 )
	    {
	      // Iterate over dofs of current element
	      tmp1 *= x(ep*method.nsize() + method.nsize() - 1);
	      for (uint n = 0; n < method.nsize(); n++)
		y(j0 + n) -= tmp1 * method.nweight(n, m);		
	    }
	  }
	  else
	  {
	    // Iterate over dofs of current element
	    tmp1 *= x(j1 + l - 1);
	    for (uint n = 0; n < method.nsize(); n++)
	      y(j0 + n) -= tmp1 * method.nweight(n, m);		
	  }
	}
      }
    }
  }
}
//-----------------------------------------------------------------------------
void MultiAdaptiveJacobian::dGmult(const uBlasVector& x, uBlasVector& y) const
{
  // Reset current sub slab
  int s0 = -1;

  // Reset elast
  for (uint i = 0; i < ts.N; i++)
    ts.elast[i] = -1;

  // Iterate over all elements
  for (uint e0 = 0; e0 < ts.ne; e0++)
  {
    // Cover all elements in current sub slab
    s0 = ts.coverNext(s0, e0);
    
    // Get element data
    const uint i0 = ts.ei[e0];
    const real a0 = ts.sa[s0];
    const real b0 = ts.sb[s0];
    const real k0 = b0 - a0;
    const uint j0 = e0 * method.nsize();
    
    // Add dependency on predecessor for all dofs of element
    const int ep = ts.ee[e0];
    if ( ep != -1 )
    {
      const real xp = x(ep*method.nsize() + method.nsize() - 1);
      for (uint n = 0; n < method.nsize(); n++)
	y(j0 + n) -= xp;
    }

    // Reset Jpos
    uint Jpos = 0;

    // Iterate over dependencies for the current component
    const Array<uint>& deps = ode.dependencies[i0];
    for (uint pos = 0; pos < deps.size(); pos++)
    {
      // Get derivative
      const real dfdu = Jvalues[Jindices[i0] + pos];

      // Skip elements which have not been covered
      const uint i1 = deps[pos];
      const int e1 = ts.elast[i1];      
      if ( e1 == -1 )
      {
	// Save dependency for later
	Jlookup[Jpos++] = dfdu;
	continue;
      }

      // Skip elements with smaller time steps
      const uint s1 = ts.es[e1];
      const real b1 = ts.sb[s1];
      if ( b1 < (a0 + DOLFIN_EPS) )
      {
	// Save dependency for later
	Jlookup[Jpos++] = dfdu;
       	continue;
      }
      
      // Get first dof for other element
      const uint j1 = e1 * method.nsize();
      
      // Use fast evaluation for elements in the same sub slab
      if ( s0 == static_cast<int>(s1) )
      {
	const real tmp = k0 * dfdu;
	for (uint n = 0; n < method.nsize(); n++)
	{
	  real sum = 0.0;
	  for (uint m = 0; m < method.qsize(); m++)
	    sum += method.nweight(n, m) * x(j1 + m);
	  y(j0 + n) -= tmp * sum;
	}
      }
      else
      {
	const real a1 = ts.sa[s1];
	const real k1 = b1 - a1;
	
	// Iterate over dofs of element
	const real tmp0 = k0 * dfdu;
	for (uint n = 0; n < method.nsize(); n++)
	{
	  // Iterate over quadrature points
	  real sum = 0.0;
	  for (uint m = 0; m < method.qsize(); m++)
	  {
	    const real tau = (a0 + k0*method.qpoint(m) - a1) / k1;
	    const real tmp1 = method.nweight(n, m);
	    
	    // Iterate over dofs of other element and add dependencies
	    for (uint l = 0; l < method.nsize(); l++)
	      sum += tmp1 * method.eval(l, tau) * x(j1 + l);
	  }
	  
	  // Add dependencies
	  y(j0 + n) -= tmp0 * sum;
	}
      }
    }
    
    // At this point, we have added dependencies to the predecessor,
    // to dofs in the same sub slab and to components with larger time
    // steps. It remains to add dependencies to components with
    // smaller time steps. We need to do this by iterating over
    // quadrature points, since this is the order in which the
    // dependencies are stored.

    // Get first dependency to components with smaller time steps for element
    uint d = ts.ed[e0];
    
    // Compute number of such dependencies for each nodal point
    const uint end = ( e0 < (ts.ne - 1) ? ts.ed[e0 + 1] : ts.nd );
    const uint ndep = (end - d) / method.nsize();
    dolfin_assert(ndep * method.nsize() == (end - d));

    // Iterate over quadrature points of current element
    for (uint m = 0; m < method.qsize(); m++)
    {
      // Compute quadrature point
      const real t = a0 + k0*method.qpoint(m);

      // Iterate over dependencies
      for (uint dep = 0; dep < ndep; dep++)
      {
	// Get element data
	const uint e1 = ts.de[d++];
	const uint j1 = e1 * method.nsize();
	const uint s1 = ts.es[e1];
	//const uint i1 = ts.ei[e1];
	const real a1 = ts.sa[s1];
	const real b1 = ts.sb[s1];
	const real k1 = b1 - a1;
	const real tau = (t - a1) / k1;
	
	// We don't know how to index Jvalues here and want to avoid
	// searching, but we were clever enough to pick out the value
	// before when we had the chance... :-)
	const real dfdu = Jlookup[dep];
	//message("Looks like df_%d/du_%d = %f", i0, i1, dfdu);      
	
	// Iterate over quadrature points of other element
	const real tmp0 = k0 * dfdu;
	for (uint l = 0; l < method.qsize(); l++)
	{
	  real tmp1 = tmp0 * method.eval(l, tau) * x(j1 + l);
	  for (uint n = 0; n < method.nsize(); n++)
	    y(j0 + n) -= tmp1 * method.nweight(n, m);
	}
      }
    }
  }
}
//-----------------------------------------------------------------------------
