// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/ODE.h>
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

  // Compute total number of dependencies
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

  
  dolfin_info("Generated Jacobian data structure for %d dependencies.", sum);
}
//-----------------------------------------------------------------------------
NewJacobianMatrix::~NewJacobianMatrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewJacobianMatrix::mult(Vec x, Vec y) const
{
  // We iterate over all degrees of freedom j in the time slab and compute
  // y_j = (Ax)_j for each degree of freedom of the system. Note that this
  // implementation will probably not work with parallel vectors since we
  // use VecGetArray to access the local arrays of the vectors

  cout << "Computing product y = Ax" << endl;
  
  // Start with y = x, accounting for the derivative dF_j/dx_j = 1
  VecCopy(x, y);

  // Get data arrays from the PETSc vectors
  real* xx(0);
  real* yy(0);
  VecGetArray(x, &xx);
  VecGetArray(y, &yy);

  // Reset dof
  uint j0 = 0;

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
    
    // Add dependency on predecessor for all dofs of element
    const int ep = ts.ee[e0];
    if ( ep != -1 )
    {
      const real xp = xx[ep*method.nsize() + method.nsize() - 1];
      for (uint n = 0; n < method.nsize(); n++)
	yy[j0 + n] -= xp;
    }

    // Iterate over dependencies for the current component
    const NewArray<uint>& deps = ode.dependencies[i0];
    for (uint pos = 0; pos < deps.size(); pos++)
    {
      // Skip elements which have not been covered
      const uint i1 = deps[pos];
      const int e1 = ts.elast[i1];      
      if ( e1 == -1 )
	continue;

      // Skip elements with smaller time steps
      const uint s1 = ts.es[e1];
      const real b1 = ts.sb[s1];
      if ( b1 < (a0 + DOLFIN_EPS) )
       	continue;
      
      // Get derivative and first dof for other element
      const real dfdu = Jvalues[Jindices[i0] + pos];
      const uint j1 = e1 * method.nsize();
      
      // Use fast evaluation for elements in the same sub slab
      if ( s0 == static_cast<int>(s1) )
      {
	if ( method.type() == NewMethod::cG )
	{
	  // Add dependency to dof of initial value if any
	  const int ep = ts.ee[e1];
	  if ( ep != -1 )
	  {
	    const real x0 = xx[ep * method.nsize() + method.nsize() - 1];
	    const real tmp = k0 * dfdu * x0;
	    for (uint n = 0; n < method.nsize(); n++)
	      yy[j0 + n] -= tmp * method.nweight(n, 0);
	  }

	  // Add dependencies to internal dofs
	  const real tmp = k0 * dfdu;
	  for (uint n = 0; n < method.nsize(); n++)
	  {
	    real sum = 0.0;
	    for (uint m = 1; m < method.qsize(); m++)
	      sum += method.nweight(n, m) * xx[j1 + m];
	    yy[j0 + n] -= tmp * sum;
	  }
	}
	else
	{
	  // Add dependencies to internal dofs
	  const real tmp = k0 * dfdu;
	  for (uint n = 0; n < method.nsize(); n++)
	  {
	    real sum = 0.0;
	    for (uint m = 0; m < method.qsize(); m++)
	      sum += method.nweight(n, m) * xx[j1 + m];
	    yy[j0 + n] -= tmp * sum;
	  }
	}
      }
      else
      {
	const real a1 = ts.sa[s1];
	const real k1 = b1 - a1;
	
	if ( method.type() == NewMethod::cG )
	{
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

	      // Add dependency to dof of initial value if any
	      const int ep = ts.ee[e1];
	      if ( ep != -1 )
	      {
		const real x0 = xx[ep * method.nsize() + method.nsize() - 1];
		sum += tmp1 * method.eval(0, tau) * x0;
	      }
	      
	      // Iterate over dofs of other element and add dependencies
	      for (uint l = 0; l < method.nsize(); l++)
		sum += tmp1 * method.eval(l + 1, tau) * xx[j1 + l];
	    }
	    
	    // Add dependencies
	    yy[j0 + n] -= tmp0 * sum;
	  }
	}
	else
	{
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
		sum += tmp1 * method.eval(l, tau) * xx[j1 + l];
	    }
	    
	    // Add dependencies
	    yy[j0 + n] -= tmp0 * sum;
	  }
	}
      }
      

      /*
      
      // Get first dependency to components with smaller time steps for element
      uint d = ts.ed[e0];
      
      // Compute number of such dependencies for each nodal point
      const uint end = ( e0 < (ts.ne - 1) ? ts.ed[e0 + 1] : ts.nd );
      const uint ndep = (end - d) / method.nsize();
      dolfin_assert(ndep * method.nsize() == (end - d));

      */
      
    }
  }

  // Restor PETSc data arrays
  VecRestoreArray(x, &xx);
  VecRestoreArray(y, &yy);
}
//-----------------------------------------------------------------------------
void NewJacobianMatrix::update(real t)
{
  dolfin_info("Recomputing Jacobian matrix at t = %f.", t);
  
}
//-----------------------------------------------------------------------------
void NewJacobianMatrix::disp() const
{

  
}
//-----------------------------------------------------------------------------
