// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "ProblemPoisson.hh"
#include "EquationPoisson.hh"
#include "EquationPoisson2d.hh"

//-----------------------------------------------------------------------------
const char *ProblemPoisson::Description()
{
  if ( space_dimension == 3 )
    return "Poisson's equation (3D)";
  else
	 return "Poisson's equation (2D)";
}
//-----------------------------------------------------------------------------
void ProblemPoisson::Solve()
{
  Equation *equation;
  SparseMatrix A;
  Vector x,b;
  KrylovSolver solver;
  GlobalField u(grid,&x);
  GlobalField f(grid,"source");

  // Get the specified dimension
  if ( space_dimension == 3 )
	 equation = new EquationPoisson();
  else
	 equation = new EquationPoisson2d();
  
  // Set up the right-hand side
  equation->AttachField(0,&f);
  
  // Set up the discretiser
  Discretiser discretiser(grid,equation);
  
  // Discretise Poisson's equation
  discretiser.Assemble(&A,&b);

  // Solve the linear system
  solver.Solve(&A,&x,&b);
    
  // Save the solution
  u.SetLabel("u","temperature");
  u.Save();

  // Cleanup
  delete equation;
}
//-----------------------------------------------------------------------------
