// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Solution.h>
#include <dolfin/RHS.h>
#include <dolfin/Element.h>
#include <dolfin/ElementGroup.h>
#include <dolfin/ElementGroupList.h>
#include <dolfin/ElementGroupIterator.h>
#include <dolfin/FixedPointIteration.h>
#include <dolfin/FixedPointIteration.h>
#include <dolfin/NonStiffIteration.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NonStiffIteration::NonStiffIteration(Solution& u, RHS& f,
				     FixedPointIteration & fixpoint,
				     unsigned int maxiter,
				     real maxdiv, real maxconv, real tol) :
  Iteration(u, f, fixpoint, maxiter, maxdiv, maxconv, tol)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NonStiffIteration::~NonStiffIteration()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Iteration::State NonStiffIteration::state() const
{
  return nonstiff;
}
//-----------------------------------------------------------------------------
void NonStiffIteration::start(ElementGroupList& list)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NonStiffIteration::start(ElementGroup& group)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NonStiffIteration::start(Element& element)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NonStiffIteration::update(ElementGroupList& list)
{
  // Simple update of group list
  for (ElementGroupIterator group(list); !group.end(); ++group)
    fixpoint.iterate(*group);
}
//-----------------------------------------------------------------------------
void NonStiffIteration::update(ElementGroup& group)
{
  // Simple update of element group
  for (ElementIterator element(group); !element.end(); ++element)
    fixpoint.iterate(*element);
}
//-----------------------------------------------------------------------------
void NonStiffIteration::update(Element& element)
{
  // Simple update of element
  element.update(f);
}
//-----------------------------------------------------------------------------
void NonStiffIteration::stabilize(ElementGroupList& list,
				  const Residuals& r, unsigned int n)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NonStiffIteration::stabilize(ElementGroup& group,
				  const Residuals& r, unsigned int n)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NonStiffIteration::stabilize(Element& element, 
				  const Residuals& r, unsigned int n)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool NonStiffIteration::converged(ElementGroupList& list, 
				  Residuals& r, unsigned int n)
{
  // Convergence handled locally when the slab contains only one element group
  if ( list.size() <= 1 )
    return n > 0;

  // Compute residual
  r.r1 = r.r2;
  r.r2 = residual(list);

  // Save initial residual
  if ( n == 0 )
    r.r0 = r.r2;
  
  return r.r2 < tol & n > 0;
}
//-----------------------------------------------------------------------------
bool NonStiffIteration::converged(ElementGroup& group, 
				  Residuals& r, unsigned int n)
{
  // Convergence handled locally when the group contains only one element
  if ( group.size() == 1 )
    return n > 0;
  
  // Compute maximum residual
  r.r1 = r.r2;
  r.r2 = residual(group);
  
  // Save initial residual
  if ( n == 0 )
    r.r0 = r.r2;
  
  return r.r2 < tol & n > 0;
}
//-----------------------------------------------------------------------------
bool NonStiffIteration::converged(Element& element, 
				  Residuals& r, unsigned int n)
{
  // Compute residual
  r.r1 = r.r2;
  r.r2 = residual(element);

  // Save initial residual
  if ( n == 0 )
    r.r0 = r.r2;
  
  return r.r2 < tol & n > 0;
}
//-----------------------------------------------------------------------------
bool NonStiffIteration::diverged(ElementGroupList& list, 
				 Residuals& r, unsigned int n,
				 Iteration::State& newstate)
{
  // Make at least two iterations
  if ( n < 2 )
    return false;

  // Check if the solution converges
  if ( r.r2 < maxconv * r.r1 )
    return false;

  // Notify change of strategy
  dolfin_info("Problem appears to be stiff, trying a stabilizing time step sequence.");
  
  // Reset group list
  fixpoint.reset(list);

  // Change state
  newstate = stiff3;

  return true;
}
//-----------------------------------------------------------------------------
bool NonStiffIteration::diverged(ElementGroup& group, 
				 Residuals& r, unsigned int n,
				 Iteration::State& newstate)
{
  // Make at least two iterations
  if ( n < 2 )
    return false;

  // Check if the solution converges
  if ( r.r2 < maxconv * r.r1 )
    return false;
  
  // Notify change of strategy
  dolfin_info("Problem appears to be stiff, trying adaptive damping.");
  
  // Reset element list
  reset(group);

  // Change state
  newstate = stiff2;

  return true;
}
//-----------------------------------------------------------------------------
bool NonStiffIteration::diverged(Element& element, 
				 Residuals& r, unsigned int n,
				 Iteration::State& newstate)
{
  // Make at least two iterations
  if ( n < 2 )
    return false;

  // Check if the solution converges
  if ( r.r2 < maxconv * r.r1 )
    return false;

  // Notify change of strategy
  dolfin_info("Problem appears to be stiff, trying diagonal damping.");
  
  // Reset element
  reset(element);

  // Change state
  newstate = stiff1;

  return true;
}
//-----------------------------------------------------------------------------
void NonStiffIteration::report() const
{
  cout << "System is non-stiff, solution computed with "
       << "simple fixed point iteration." << endl;
}
//-----------------------------------------------------------------------------
