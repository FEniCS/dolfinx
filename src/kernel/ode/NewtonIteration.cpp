// Copyright (C) 2003 Johan Jansson.
// Licensed under the GNU GPL Version 2.

#include <iostream>

#include <cmath>
#include <dolfin/dolfin_math.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/Solution.h>
#include <dolfin/RHS.h>
#include <dolfin/Element.h>
#include <dolfin/ElementGroup.h>
#include <dolfin/ElementGroupList.h>
#include <dolfin/ElementGroupIterator.h>
#include <dolfin/FixedPointIteration.h>
#include <dolfin/KrylovSolver.h>
#include <dolfin/NewtonIteration.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewtonIteration::NewtonIteration(Solution& u, RHS& f,
				 FixedPointIteration& fixpoint,
				 unsigned int maxiter,
				 real maxdiv, real maxconv, real tol,
				 unsigned int depth) :
  Iteration(u, f, fixpoint, maxiter, maxdiv, maxconv, tol, depth), 
  J(f), t0(0), njac(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewtonIteration::~NewtonIteration()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Iteration::State NewtonIteration::state() const
{
  return newton;
}
//-----------------------------------------------------------------------------
void NewtonIteration::start(ElementGroupList& list)
{
  // Get the first element
  ElementIterator element(list);
  t0 = element->starttime();
  
  // Compute Jacobian at current time
  J.update(t0);
  njac++;

  // Initialize data
  unsigned int dof = J.update(list);
  r.init(dof);
  dx.init(dof);
  
  //J.show();
}
//-----------------------------------------------------------------------------
void NewtonIteration::start(ElementGroup& group)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewtonIteration::start(Element& element)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewtonIteration::update(ElementGroupList& list, Increments& d)
{
dolfin_error("This function needs to be updated to the new format.");

/*
  // Compute element residuals for all elements
  unsigned int dof = 0;
  for (ElementIterator element(list); !element.end(); ++element)
  {
    element->computeElementResidual(f, r.values + dof);
    dof += element->size();
  }
  
  // Solve linear system
  //dolfin_log(false);
  KrylovSolver solver;
  solver.solve(J, dx, r);
  //dolfin_log(true);

  cout << "dx = ";
  dx.show();
  
  // Set values
  dof = 0;
  for (ElementIterator element(list); !element.end(); ++element)
  {
    // Subtract increment
    element->sub(dx.values + dof);

    // Update initial data, using r to store the data
    real& u0 = r(element->index());
    if ( element->starttime() != t0 )
      element->update(u0);
    u0 = element->endval();

    // Jump to next degree of freedom
    dof += element->size();
  }

  cout << "Increment: " << dx.norm() << endl;

  // l2 norm of increment
  d = dx.norm();
*/
}
//-----------------------------------------------------------------------------
void NewtonIteration::update(ElementGroup& group, Increments& d)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewtonIteration::update(Element& element, Increments& d)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewtonIteration::stabilize(ElementGroupList& list,
				  const Increments& d, unsigned int n)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewtonIteration::stabilize(ElementGroup& group,
				 const Increments& d,  unsigned int n)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewtonIteration::stabilize(Element& element,
				  const Increments& d, unsigned int n)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool NewtonIteration::converged(ElementGroupList& list,
				const Increments& d, unsigned int n)
{
  // First check increment
  if ( d.d2 > tol || n == 0 )
    return false;
  
  // If increment is small, then check residual
  return residual(list) < tol;
}
//-----------------------------------------------------------------------------
bool NewtonIteration::converged(ElementGroup& group,
				  const Increments& d, unsigned int n)
{
  return d.d2 < tol && n > 0;
}
//-----------------------------------------------------------------------------
bool NewtonIteration::converged(Element& element,
				  const Increments& d, unsigned int n)
{
  return d.d2 < tol && n > 0;
}
//-----------------------------------------------------------------------------
bool NewtonIteration::diverged(ElementGroupList& list, const Increments& d,
				 unsigned int n, State& newstate)
{
  return false;
}
//-----------------------------------------------------------------------------
bool NewtonIteration::diverged(ElementGroup& group, const Increments& d,
				 unsigned int n, State& newstate)
{
  return false;
}
//-----------------------------------------------------------------------------
bool NewtonIteration::diverged(Element& element, const Increments& d,
				 unsigned int n, State& newstate)
{
  return false;
}
//-----------------------------------------------------------------------------
void NewtonIteration::report() const
{
  cout << "System is stiff, solution computed with "
       << "Newton's method." << endl;
  cout << "Computed the Jacobian " << njac << " times." << endl;
}
//-----------------------------------------------------------------------------
