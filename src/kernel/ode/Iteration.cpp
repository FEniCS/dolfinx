// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/dolfin_math.h>
#include <dolfin/Solution.h>
#include <dolfin/RHS.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/Element.h>
#include <dolfin/ElementGroup.h>
#include <dolfin/ElementGroupList.h>
#include <dolfin/Iteration.h>

using namespace dolfin;

// FIXME: make magic numbers (0.75, 0.1) parameters

//-----------------------------------------------------------------------------
Iteration::Iteration(Solution& u, RHS& f, FixedPointIteration& fixpoint,
		     unsigned int maxiter, real maxdiv, real maxconv, real tol,
		     unsigned int depth) : 
  u(u), f(f), fixpoint(fixpoint), 
  maxiter(maxiter), maxdiv(maxdiv), maxconv(maxconv), tol(tol),
  alpha(1), gamma(1.0/sqrt(2.0)), r0(0), m(0), j(0), _depth(depth)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Iteration::~Iteration()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
unsigned int Iteration::depth() const
{
  return _depth;
}
//-----------------------------------------------------------------------------
void Iteration::down()
{
  ++_depth;
}
//-----------------------------------------------------------------------------
void Iteration::up()
{
  --_depth;
}
//-----------------------------------------------------------------------------
void Iteration::init(ElementGroup& group)
{
  // Update initial data for elements
  for (ElementIterator element(group); !element.end(); ++element)
    init(*element);
}
//-----------------------------------------------------------------------------
void Iteration::init(Element& element)
{
  // Get initial value
  real u0 = u(element.index(), element.starttime());
  
  // Reset element
  element.update(u0);
}
//-----------------------------------------------------------------------------
void Iteration::reset(ElementGroupList& list)
{
  // Reset all elements
  for (ElementIterator element(list); !element.end(); ++element)
    reset(*element);
}
//-----------------------------------------------------------------------------
void Iteration::reset(ElementGroup& group)
{
  // Reset all elements
  for (ElementIterator element(group); !element.end(); ++element)
    reset(*element);
}
//-----------------------------------------------------------------------------
void Iteration::reset(Element& element)
{
  // Get initial value
  real u0 = u(element.index(), element.starttime());

  // Reset element
  element.set(u0);
}
//-----------------------------------------------------------------------------
real Iteration::residual(ElementGroupList& list)
{
  real r = 0.0;
  for (ElementIterator element(list); !element.end(); ++element)
    r += sqr(residual(*element));

  return sqrt(r);
}
//-----------------------------------------------------------------------------
real Iteration::residual(ElementGroup& group)
{
  real r = 0.0;
  for (ElementIterator element(group); !element.end(); ++element)
    r += sqr(residual(*element));

  return sqrt(r);
}
//-----------------------------------------------------------------------------
real Iteration::residual(Element& element)
{
  return fabs(element.computeElementResidual(f));
}
//-----------------------------------------------------------------------------
bool Iteration::stabilize(const Residuals& r, unsigned int n)
{
  // Take action depending on j, the remaining number of iterations
  // with small alpha.
  //
  //   j = 0 : increasing alpha (or alpha = 1)
  //   j = 1 : last stabilizing iteration
  //   j > 1 : still stabilizing

  // FIXME: changed here
  // Make at least one iteration before stabilizing
  if ( n < 2 )
    return false;
  
  switch ( j ) {
  case 0:
    // Increase alpha with a factor 2 towards alpha = 1
    if ( r.r2 > maxconv*r.r1 )
      alpha = 2.0 * alpha / (1.0 + 2.0*alpha);
    break;
  case 1:
    // Continue with another round of stabilizing steps if it seems to work
    if ( pow(r.r2/r0, 1.0/static_cast<real>(m)) < 0.75 )
    {
      // Double the number of stabilizing iterations
      m *= 2;
      j = m;
      
      // Choose a slightly larger alpha if convergence is monotone
      if ( r.r2 < 0.75*r.r1 && r.r1 < 0.75*r0 )
	alpha *= 1.1;
      
      // Save residual at start of stabilizing iterations
      r0 = r.r2;
    }
    else
    {
      // Finish stabilization
      j = 0;
    }
    break;
  default:
    // Decrease number of remaining iterations with small alpha
    j -= 1;
  }

  // Check if stabilization is needed
  return r.r2 > r.r1 && j == 0;
}
//-----------------------------------------------------------------------------
real Iteration::computeDivergence(ElementGroupList& list, const Residuals& r)
{
  cout << "Computing divergence for time slab" << endl;
  
  // Successive residuals
  real r1 = r.r1;
  real r2 = r.r2;

  // Successive convergence factors
  real rho2 = r2 / r1;
  real rho1 = rho2;

  // Save current alpha and change alpha to 1 for divergence computation
  real alpha0 = alpha;
  alpha = 1.0;

  // Save solution values before iteration
  initData(x0, x1.size);
  copyData(list, x0);

  for (unsigned int n = 0; n < maxiter; n++)
  {
    // Update time slab
    update(list);
    
    // Compute residual
    r1 = r2;
    r2 = residual(list);
  
    // Compute divergence
    rho1 = rho2;
    rho2 = r2 / (DOLFIN_EPS + r1);

    cout << "  rho = " << rho2 << endl;
    
    // Check if the divergence factor has converged
    if ( abs(rho2-rho1) < 0.1 * rho1 )
    {
      dolfin_debug1("Computed divergence rate in %d iterations", n + 1);
      break;
    }
    
  }

  // Restore alpha
  alpha = alpha0;

  // Restore solution values
  copyData(x0, list);

  // Restore initial data for all elements
  for (ElementIterator element(list); !element.end(); ++element)
    init(*element);

  return rho2;
}
//-----------------------------------------------------------------------------
real Iteration::computeDivergence(ElementGroup& group, const Residuals& r)
{
  // Successive residuals
  real r1 = r.r1;
  real r2 = r.r2;
  
  // Successive convergence factors
  real rho2 = r2 / r1;
  real rho1 = rho2;
  
  // Save current alpha and change alpha to 1 for divergence computation
  real alpha0 = alpha;
  alpha = 1.0;

  // Save solution values before iteration
  initData(x0, x1.size);
  copyData(group, x0);

  for (unsigned int n = 0; n < maxiter; n++)
  {
    // Update element group
    update(group);
    
    // Compute residual
    r1 = r2;
    r2 = residual(group);
  
    // Compute divergence
    rho1 = rho2;
    rho2 = r2 / (DOLFIN_EPS + r1);

    cout << "  rho = " << rho2 << endl;

    // Check if the divergence factor has converged
    if ( abs(rho2-rho1) < 0.1 * rho1 )
    {
      dolfin_debug1("Computed divergence rate in %d iterations", n + 1);
      break;
    }
    
  }

  // Restore alpha
  alpha = alpha0;

  // Restore solution values
  copyData(x0, group);

  return rho2;
}
//-----------------------------------------------------------------------------
real Iteration::computeAlpha(real rho) const
{
  return gamma / (1.0 + rho);
}
//-----------------------------------------------------------------------------
unsigned int Iteration::computeSteps(real rho) const
{
  return ceil_int(1.0 + log(rho) / log(1.0/(1.0-gamma*gamma)));
}
//-----------------------------------------------------------------------------
void Iteration::initInitialData(real t0)
{
  // Allocate values
  unsigned int N = u.size();
  initData(u0, N);

  // Set initial values for all components
  for (unsigned int i = 0; i < N; i++)
    u0.values[i] = u(i, t0);
}
//-----------------------------------------------------------------------------
void Iteration::initData(Values& values, unsigned int size)
{
  // Reallocate data if necessary
  if ( size > values.size )
    values.init(size);

  // Reset offset
  values.offset = 0;
}
//-----------------------------------------------------------------------------
unsigned int Iteration::dataSize(ElementGroupList& list)
{
  // Compute total number of values
  int size = 0;
  for (ElementIterator element(list); !element.end(); ++element)
    size += element->size();
  
  return size;
}
//-----------------------------------------------------------------------------
unsigned int Iteration::dataSize(ElementGroup& group)
{
  // Compute total number of values
  int size = 0;  
  for (ElementIterator element(group); !element.end(); ++element)
    size += element->size();
  
  return size;
}
//-----------------------------------------------------------------------------
void Iteration::copyData(ElementGroupList& list, Values& values)
{
  // Copy data from group list
  unsigned int offset = 0;
  for (ElementIterator element(list); !element.end(); ++element)
  {
    // Copy values from element
    element->get(values.values + offset);

    // Increase offset
    offset += element->size();
  }
}
//-----------------------------------------------------------------------------
void Iteration::copyData(Values& values, ElementGroupList& list)
{
  // Copy data to group list
  unsigned int offset = 0;
  for (ElementIterator element(list); !element.end(); ++element)
  {
    // Copy values to element
    element->set(values.values + offset);

    // Increase offset
    offset += element->size();
  }
}
//-----------------------------------------------------------------------------
void Iteration::copyData(ElementGroup& group, Values& values)
{
  // Copy data from element group
  unsigned int offset = 0;
  for (ElementIterator element(group); !element.end(); ++element)
  {
    // Copy values from element
    element->get(values.values + offset);

    // Increase offset
    offset += element->size();
  }
}
//-----------------------------------------------------------------------------
void Iteration::copyData(Values& values, ElementGroup& group)
{
  // Copy data to element group
  unsigned int offset = 0;
  for (ElementIterator element(group); !element.end(); ++element)
  {
    // Copy values to element
    element->set(values.values + offset);

    // Increase offset
    offset += element->size();
  }
}
//-----------------------------------------------------------------------------
// Iteration::Values
//-----------------------------------------------------------------------------
Iteration::Values::Values() : values(0), size(0), offset(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Iteration::Values::~Values()
{
  if ( values )
    delete [] values;
  values = 0;
}
//-----------------------------------------------------------------------------
void Iteration::Values::init(unsigned int size)
{
  dolfin_assert(size > 0);

  if ( values )
    delete [] values;
  
  values = new real[size];
  dolfin_assert(values);
  this->size = size;
  offset = 0;
}
//-----------------------------------------------------------------------------
