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
#include <dolfin/NewtonIteration.h>

// FIXME: Use GMRES::solve() instead
#include <dolfin/KrylovSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewtonIteration::NewtonIteration(Solution& u, RHS& f,
				     FixedPointIteration& fixpoint,
				     unsigned int maxiter,
				     real maxdiv, real maxconv, real tol,
				     unsigned int depth) :
  Iteration(u, f, fixpoint, maxiter, maxdiv, maxconv, tol, depth)
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
  // Do nothing
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

  // Decide when to do J.update(). Only compute a new Jacobian when
  // necessary. Perhaps we can keep the Jacobian for some time. At
  // least for one time slab? Use GMRES::solve(J, dx, F) to solve.


  // Assume only dG0 for now

  cout << "Newton slab" << endl;

  // Count degrees of freedom

  int n = 0;
  for (ElementIterator element(list); !element.end(); ++element)
  {
    n++;
  }  

  //cout << "dof: " << n << endl;

  Matrix J(n, n);
  Vector res(n);
  Vector sk(n);

  int i = 0;
  for (ElementIterator element(list); !element.end(); ++element)
  {
    //cout << "u0: " << u0.values[element->index()] << endl;

    // Update initial value for element
    //element->update(u0.values[element->index()]);

    res(i) = element->computeDiscreteResidual(f);

    i++;
  }

  //res.show();

  // Compute Jacobian

  i = 0;
  for (ElementIterator element1(list); !element1.end(); ++element1)
  {
    double h = 1e-12;

    double origval;
    double pertval;
    element1->get(&origval);

    pertval = origval + h;
    element1->set(&pertval);

    int j = 0;
    for (ElementIterator element2(list); !element2.end(); ++element2)
    {
      double res1;
      double res2;

      res1 = element2->computeDiscreteResidual(f);
      res2 = res(j);

      J(i, j) = (res1 - res2) / h;

      j++;
    }

    element1->set(&origval);
    i++;
  }  

  //J.show();

  res *= -1.0;

  KrylovSolver solver(KrylovSolver::GMRES);
  solver.solve(J, sk, res);

  i = 0;
  for (ElementIterator element(list); !element.end(); ++element)
  {
    double newval;

    element->get(&newval);
    newval += sk(i);
    element->set(&newval);

    i++;
  }

  // Update initial data for all elements
  for (ElementIterator element(list); !element.end(); ++element)
    init(*element);

}
//-----------------------------------------------------------------------------
void NewtonIteration::update(ElementGroup& group, Increments& d)
{
  cout << "Newton group" << endl;

  // Assume only dG0 for now

  // Count degrees of freedom

  int n = 0;
  for (ElementIterator element(group); !element.end(); ++element)
  {
    n++;
  }  

  //cout << "dof: " << n << endl;

  Matrix J(n, n);
  Vector res(n);
  Vector sk(n);

  int i = 0;
  for (ElementIterator element(group); !element.end(); ++element)
  {
    //cout << "u0: " << u0.values[element->index()] << endl;

    // Update initial value for element
    //element->update(u0.values[element->index()]);

    res(i) = element->computeDiscreteResidual(f);

    i++;
  }

  //res.show();

  // Compute Jacobian

  i = 0;
  for (ElementIterator element1(group); !element1.end(); ++element1)
  {
    double h = 1e-8;

    double origval;
    double pertval;
    element1->get(&origval);

    pertval = origval + h;
    element1->set(&pertval);

    int j = 0;
    for (ElementIterator element2(group); !element2.end(); ++element2)
    {
      double res1;
      double res2;

      res1 = element2->computeDiscreteResidual(f);
      res2 = res(j);

      J(i, j) = (res1 - res2) / h;

      j++;
    }

    element1->set(&origval);
    i++;
  }  

  //J.show();

  res *= -1.0;

  KrylovSolver solver(KrylovSolver::GMRES);
  solver.solve(J, sk, res);

  i = 0;
  for (ElementIterator element(group); !element.end(); ++element)
  {
    double newval;

    element->get(&newval);
    newval += sk(i);
    element->set(&newval);

    i++;
  }

  // Update initial data for all elements
  for (ElementIterator element(group); !element.end(); ++element)
    init(*element);


}
//-----------------------------------------------------------------------------
void NewtonIteration::update(Element& element, Increments& d)
{
  cout << "Newton element" << endl;
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
  //if ( d.d2 > tol || n == 0 )
  //return false;

  // If increment is small, then check residual
  return residual(list) < tol;
}
//-----------------------------------------------------------------------------
bool NewtonIteration::converged(ElementGroup& group,
				  const Increments& d, unsigned int n)
{
  //return d.d2 < tol && n > 0;
  return residual(group) < tol && n > 0;
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
}
//-----------------------------------------------------------------------------
