// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <iostream>

#include <dolfin/dolfin_math.h>
#include <dolfin/ODE.h>
#include <dolfin/NewVector.h>
#include <dolfin/NewTimeSlab.h>
#include <dolfin/NewMethod.h>
#include <dolfin/NewJacobianMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewJacobianMatrix::NewJacobianMatrix(ODE& ode, NewTimeSlab& timeslab,
				     const NewMethod& method) :
  ode(ode), ts(timeslab), method(method), Jvalues(0), Jindices(0)
{
  // Allocate Jacobian row indices
  Jindices = new uint[ode.size()];

  // Compute total and maximum number of dependencies
  uint sum = 0;
  uint maxsize = 0;
  for (uint i = 0; i < ode.size(); i++)
  {
    Jindices[i] = sum;

    const uint size = ode.dependencies[i].size();
    sum += size;
    if ( size > maxsize )
      maxsize = size;
  }

  // Allocate Jacobian values
  Jvalues = new real[sum];
  for (uint pos = 0; pos < sum; pos++)
    Jvalues[pos] = 0.0;

  // Allocate lookup table for dependencies to components with small time steps
  Jlookup = new real[max(1, maxsize - 1)];
  
  dolfin_info("Generated Jacobian data structure for %d dependencies.", sum);
}
//-----------------------------------------------------------------------------
NewJacobianMatrix::~NewJacobianMatrix()
{
  delete [] Jindices;
  delete [] Jvalues;
  delete [] Jlookup;
}
//-----------------------------------------------------------------------------
void NewJacobianMatrix::mult(Vec x, Vec y) const
{
  // We iterate over all degrees of freedom j in the time slab and compute
  // y_j = (Ax)_j for each degree of freedom of the system. Note that this
  // implementation will probably not work with parallel vectors since we
  // use VecGetArray to access the local arrays of the vectors

  std::cout << "pointer before: " << x << std::endl;
  NewVector xxx(x);
  x = xxx.vec();
  std::cout << "pointer after:  " << x << std::endl;

  // Start with y = x, accounting for the derivative dF_j/dx_j = 1
  VecCopy(x, y);

  // Get data arrays from the PETSc vectors
  real* xx(0);
  real* yy(0);
  VecGetArray(x, &xx);
  VecGetArray(y, &yy);

  // Choose method
  if ( method.type() == NewMethod::cG )
    cGmult(xx, yy);
  else
    dGmult(xx, yy);

  // Restore PETSc data arrays
  VecRestoreArray(x, &xx);
  VecRestoreArray(y, &yy);
}
//-----------------------------------------------------------------------------
void NewJacobianMatrix::update()
{
  // Compute Jacobian at the beginning of the slab
  real t = ts.starttime();
  dolfin_info("Recomputing Jacobian matrix at t = %f.", t);

  // Update vector u to values at the left end-point
  for (uint i = 0; i < ode.size(); i++)
    ts.u[i] = ts.u0[i];
 
  // Compute Jacobian
  for (uint i = 0; i < ode.size(); i++)
  {
    const NewArray<uint>& deps = ode.dependencies[i];
    for (uint pos = 0; pos < deps.size(); pos++)
      Jvalues[Jindices[i] + pos] = ode.dfdu(ts.u, t, i, deps[pos]);
  }
}
//-----------------------------------------------------------------------------
void NewJacobianMatrix::cGmult(real x[], real y[]) const
{
  // Reset current sub slab
  int s0 = -1;

  // Reset elast
  ts.elast = -1;

  // Iterate over all elements
  for (uint e0 = 0; e0 < ts.ne; e0++)
  {
    // Cover all elements in current sub slab
    s0 = ts.cover(s0, e0);
    
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
      const real xp = x[ep*method.nsize() + method.nsize() - 1];
      for (uint n = 0; n < method.nsize(); n++)
	y[j0 + n] -= xp;
    }

    // Reset Jpos
    uint Jpos = 0;

    // Iterate over dependencies for the current component
    const NewArray<uint>& deps = ode.dependencies[i0];
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
	const real tmp = k0 * dfdu * x[e1 * method.nsize() + method.nsize() - 1];
	for (uint n = 0; n < method.nsize(); n++)
	  y[j0 + n] -= tmp * method.nweight(n, 0);

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
	  const real tmp1 = tmp0 * x[ep * method.nsize() + method.nsize() - 1];
	  for (uint n = 0; n < method.nsize(); n++)
	    y[j0 + n] -= tmp1 * method.nweight(n, 0);
	}
	
	// Add dependencies to internal dofs
	for (uint n = 0; n < method.nsize(); n++)
	{
	  real sum = 0.0;
	  for (uint m = 0; m < method.nsize(); m++)
	    sum += method.nweight(n, m + 1) * x[j1 + m];
	  y[j0 + n] -= tmp0 * sum;
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
	      const real x0 = x[ep * method.nsize() + method.nsize() - 1];
	      sum += tmp1 * method.eval(0, tau) * x0;
	    }
	    
	    // Iterate over dofs of other element and add dependencies
	    for (uint l = 0; l < method.nsize(); l++)
	      sum += tmp1 * method.eval(l + 1, tau) * x[j1 + l];
	  }
	    
	  // Add dependencies
	  y[j0 + n] -= tmp0 * sum;
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
	//dolfin_info("Looks like df_%d/du_%d = %f", i0, i1, dfdu);      
	
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
	      tmp1 *= x[ep*method.nsize() + method.nsize() - 1];
	      for (uint n = 0; n < method.nsize(); n++)
		y[j0 + n] -= tmp1 * method.nweight(n, m);		
	    }
	  }
	  else
	  {
	    // Iterate over dofs of current element
	    tmp1 *= x[j1 + l - 1];
	    for (uint n = 0; n < method.nsize(); n++)
	      y[j0 + n] -= tmp1 * method.nweight(n, m);		
	  }
	}
      }
    }
  }
}
//-----------------------------------------------------------------------------
void NewJacobianMatrix::dGmult(real x[], real y[]) const
{
  // Reset current sub slab
  int s0 = -1;

  // Reset elast
  ts.elast = -1;

  // Iterate over all elements
  for (uint e0 = 0; e0 < ts.ne; e0++)
  {
    // Cover all elements in current sub slab
    s0 = ts.cover(s0, e0);
    
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
      const real xp = x[ep*method.nsize() + method.nsize() - 1];
      for (uint n = 0; n < method.nsize(); n++)
	y[j0 + n] -= xp;
    }

    // Reset Jpos
    uint Jpos = 0;

    // Iterate over dependencies for the current component
    const NewArray<uint>& deps = ode.dependencies[i0];
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
	    sum += method.nweight(n, m) * x[j1 + m];
	  y[j0 + n] -= tmp * sum;
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
	      sum += tmp1 * method.eval(l, tau) * x[j1 + l];
	  }
	  
	  // Add dependencies
	  y[j0 + n] -= tmp0 * sum;
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
	//dolfin_info("Looks like df_%d/du_%d = %f", i0, i1, dfdu);      
	
	// Iterate over quadrature points of other element
	const real tmp0 = k0 * dfdu;
	for (uint l = 0; l < method.qsize(); l++)
	{
	  real tmp1 = tmp0 * method.eval(l, tau) * x[j1 + l];
	  for (uint n = 0; n < method.nsize(); n++)
	    y[j0 + n] -= tmp1 * method.nweight(n, m);
	}
      }
    }
  }
}
//-----------------------------------------------------------------------------
