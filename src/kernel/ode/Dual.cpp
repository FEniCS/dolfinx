// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/Function.h>
#include <dolfin/Dual.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Dual::Dual(ODE& primal, Function& u) : 
  ODE(primal.size()), primal(primal), u(u), buffer(primal.size())
{
  // Set sparsity to transpose of sparsity for the primal
  sparsity.transp(primal.sparsity);

  // Set end time
  T = primal.endtime();
}
//-----------------------------------------------------------------------------
Dual::~Dual()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real Dual::u0(unsigned int i)
{
  return 1.0 / sqrt(static_cast<real>(N));
}
//-----------------------------------------------------------------------------
real Dual::f(const Vector& phi, real t, unsigned int i)
{
  // FIXME: Here we can do some optimization. Since we compute the sum
  // FIXME: over all dual dependencies of i we will do the update of
  // FIXME: buffer values many times. Since there is probably some overlap
  // FIXME: we could precompute a new sparsity pattern taking all these
  // FIXME: dependencies into account and then updating the buffer values
  // FIXME: outside the sum.

  real sum = 0.0;  
  for (Sparsity::Iterator j(i, sparsity); !j.end(); ++j)
    sum += dFdU(j, i, T - t) * phi(j);
  
  return sum;
}
//-----------------------------------------------------------------------------
real Dual::dFdU(unsigned int i, unsigned int j, real t)
{
  // Get values of u
  for (Sparsity::Iterator it(i, primal.sparsity); !it.end(); ++it)
    buffer(it) = u(it, t);
  
  // Small change in u_j
  double dU = DOLFIN_SQRT_EPS * abs(buffer(j));
  if ( dU == 0.0 )
    dU = DOLFIN_SQRT_EPS;

  // Save value of u_j
  real uj = buffer(j);

  // F values
  buffer(j) -= 0.5 * dU;
  real f1 = primal.f(buffer, t, i);
  
  buffer(j) = uj + 0.5*dU;
  real f2 = primal.f(buffer, t, i);
         
  // Compute derivative
  if ( abs(f1-f2) < DOLFIN_EPS * max(abs(f1),abs(f2)) )
    return 0.0;

  cout << "Dual: " << fabs(f2-f1)/dU << endl;

  return (f2-f1) / dU;
}
//-----------------------------------------------------------------------------
