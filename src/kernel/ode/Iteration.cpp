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
  u(u), f(f), fixpoint(fixpoint), maxiter(maxiter), maxdiv(maxdiv),
  maxconv(maxconv), tol(tol), alpha(1), gamma(1.0/sqrt(2.0)), d0(0), m(0),
  j(0), _depth(depth), _accept(true)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Iteration::~Iteration()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Iteration::stabilization(real& alpha, unsigned int& m) const
{
  alpha = this->alpha;
  m = this->m;
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
  real u0 = u(element.index(), 0, element.starttime());
  
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
  real u0 = u(element.index(), 0, element.starttime());

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
bool Iteration::accept() const
{
  return _accept;
}
//-----------------------------------------------------------------------------
bool Iteration::stabilize(const Increments& d, unsigned int n, real r)
{
  // Take action depending on j, the remaining number of iterations
  // with small alpha.
  //
  //   j = 0 : increasing alpha (or alpha = 1)
  //   j = 1 : last stabilizing iteration
  //   j > 1 : still stabilizing

  // Make at least one iteration before stabilizing
  if ( n < 1 )
    return false;
  
  cout << "j = " << j << endl;

  switch ( j ) {
  case 0:
    // Increase alpha with a factor 2 towards alpha = 1
    if ( d.d2 > maxconv*d.d1 )
      alpha = 2.0 * alpha / (1.0 + alpha);
    break;
  case 1:
    // Continue with another round of stabilizing steps if it seems to work
    if ( pow(d.d2/d0, 1.0/static_cast<real>(m)) < 0.75 )
    {
      // Double the number of stabilizing iterations
      m *= 2;
      j = m;

      // Choose a slightly larger alpha if convergence is monotone
      if ( d.d2 < 0.75*d.d1 && d.d1 < 0.75*d0 )
      {
	cout << "Increasing alpha by a factor 1.1" << endl;
	alpha *= 1.1;
      }      

      // Save increment at start of stabilizing iterations
      d0 = d.d2;
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

  if ( r > d.d2 )
    cout << "Seems to be diverging (residual condition): " << d << endl;

  if ( d.d2 > d.d1 )
    cout << "Seems to be diverging (increment condition): " << d << endl;

  // Assume that we should accept the solution
  _accept = true;
  
  // Check if stabilization is needed
  if ( d.d2 > d.d1 && d.d1 > d0 )
    return true;

  if ( d.d2 > d.d1 && j == 0 )
    return true;

  if ( r > d.d2 && j == 0 )
  {
    _accept = false;
    return true;
  }
  
  return false;
}
//-----------------------------------------------------------------------------
real Iteration::computeDivergence(ElementGroupList& list)
{
  // Increments for iteration
  Increments d;
  
  // Successive convergence factors
  real rho1 = 1.0;
  real rho2 = 1.0;
  
  // Save solution values before iteration
  if ( _accept )
  {
    initData(x0, x1.size);
    copyData(list, x0);
  }

  for (unsigned int n = 0; n < maxiter; n++)
  {
    // Update element group
    updateUnstabilized(list, d);

    // Check for convergence
    if ( d.d2 < tol )
      return 1.0;
    
    // Make at least 2 iterations
    if ( n < 1 )
      continue;

    // Compute divergence (cumulative geometric mean value)
    rho1 = rho2;
    real nn = std::max(1.0, static_cast<real>(n));
    real rhonew = d.d2 / (DOLFIN_EPS + d.d1);
    rho2 = pow(rho2, (nn-1.0)/nn) * pow(rhonew, 1.0/nn);

    // Check the computed convergence rate 
    if ( !positive(rho2) )
    {
      rho2 = 2.0 / alpha;
      break;
    }

    // Check if the divergence factor has converged
    if ( abs(rho2-rho1) < 0.1 * rho1 )
      break;
  }

  // Restore solution values
  if ( _accept )
    copyData(x0, list);

  return std::max(2.0, rho2);
}
//-----------------------------------------------------------------------------
real Iteration::computeDivergence(ElementGroup& group)
{
  // Increments for iteration
  Increments d;
  
  // Successive convergence factors
  real rho1 = 1.0;
  real rho2 = 1.0;
  
  // Save solution values before iteration
  if ( _accept )
  {
    initData(x0, x1.size);
    copyData(group, x0);
  }

  for (unsigned int n = 0; n < maxiter; n++)
  {
    // Update element group
    updateUnstabilized(group, d);

    // Check for convergence
    if ( d.d2 < tol )
      return 1.0;
    
    // Make at least 2 iterations
    if ( n < 1 )
      continue;

    // Compute divergence (cumulative geometric mean value)
    rho1 = rho2;
    real nn = std::max(1.0, static_cast<real>(n));
    real rhonew = d.d2 / (DOLFIN_EPS + d.d1);
    rho2 = pow(rho2, (nn-1.0)/nn) * pow(rhonew, 1.0/nn);

    // Check the computed convergence rate 
    if ( !positive(rho2) )
    {
      rho2 = 2.0 / alpha;
      break;
    }

    // Check if the divergence factor has converged
    if ( abs(rho2-rho1) < 0.1 * rho1 )
      break;
  }

  // Restore solution values
  if ( _accept )
    copyData(x0, group);

  return std::max(2.0, rho2);
}
//-----------------------------------------------------------------------------
void Iteration::updateUnstabilized(ElementGroupList& list, Increments& d)
{
// Reset values
  x1.offset = 0;
  
  // Iterate on each element and compute the l2 norm of the increments
  real increment = 0.0;
  for (ElementIterator element(list); !element.end(); ++element)
  {
    real di = fabs(element->update(f, x1.values + x1.offset));
    increment += di*di;
    x1.offset += element->size();
  }
  d = sqrt(increment);

  // Copy values to elements
  copyData(x1, list);
}
//-----------------------------------------------------------------------------
void Iteration::updateUnstabilized(ElementGroup& group, Increments& d)
{
  // Reset values
  x1.offset = 0;
  
  // Iterate on each element and compute the l2 norm of the increments
  real increment = 0.0;
  for (ElementIterator element(group); !element.end(); ++element)
  {
    real di = fabs(element->update(f, x1.values + x1.offset));
    increment += di*di;
    x1.offset += element->size();
  }
  d = sqrt(increment);

  // Copy values to elements
  copyData(x1, group);
}
//-----------------------------------------------------------------------------
real Iteration::computeAlpha(real rho) const
{
  return gamma / (1.0 + rho);
}
//-----------------------------------------------------------------------------
unsigned int Iteration::computeSteps(real rho) const
{
  dolfin_assert(rho >= 1.0);
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
    u0.values[i] = u(i, 0, t0);
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
  cout << "  Saving data" << endl;
  
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
bool Iteration::positive(real number) const
{
  return number >= 0.0 && number < std::numeric_limits<real>::max();
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
