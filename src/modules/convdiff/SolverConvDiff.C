// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "ProblemConvDiff.hh"
#include "EquationConvDiff_cG1dG0.hh"
#include "EquationConvDiff2d_cG1dG0.hh"
#include <stdlib.h>

//-----------------------------------------------------------------------------
const char *ProblemConvDiff::Description()
{
  if ( space_dimension == 2 )
	 return "Convection-Diffusion equation (2D)";
  else
	 return "Convection-Diffusion equation (3D)";
}
//-----------------------------------------------------------------------------
void ProblemConvDiff::solve()
{
  Equation *equation;
  SparseMatrix A;
  Vector b;
  Vector u(no_nodes);
  Vector up(no_nodes);
  KrylovSolver solver;
  real T0,T;
  
  // Choose equation
  if ( space_dimension == 2 )
	 equation = new EquationConvDiff2d_cG1dG0();
  else
	 equation = new EquationConvDiff_cG1dG0();
  
  // Global fields
  GlobalField Up(grid,&up);
  GlobalField U(grid,&u);
  GlobalField f(grid,"source");
  GlobalField eps(grid,"diffusivity");
  GlobalField bx(grid,"x-convection");
  GlobalField by(grid,"y-convection");
  GlobalField bz(grid,"z-convection");
  
  // Attach fields to equation
  equation->AttachField(0,&Up);
  equation->AttachField(1,&eps);
  equation->AttachField(2,&f);
  equation->AttachField(3,&bx);
  equation->AttachField(4,&by);
  if ( space_dimension == 3 )
	 equation->AttachField(5,&bz);
  
  // Set up the discretiser
  Discretiser discretiser(grid,equation);

  // Prepare time stepping
  settings->Get("start time",&T0);
  settings->Get("final time",&T);
  int time_step = 0;
  real t        = T0;
  real dt       = grid->GetSmallestDiameter();
  
  // Save initial value
  U.SetLabel("u","Temperature");
  U.Save(t);
  
  // Start time stepping
  while (t < T){

    time_step++;
	 t += dt;
    
    // Set solution to previous solution
    up.CopyFrom(&u);

	 // Write info
    display->Progress(0,(t-T0)/(T-T0),"i=%i t=%f",time_step,t);
    display->Message(0,"||u0||= %f ||u1|| = %f",up.Norm(),u.Norm());

	 // Set time and time-step
	 equation->SetTime(t);
	 equation->SetTimeStep(dt);
	 
    // Discretise equation
    discretiser.Assemble(&A,&b);
	 
    // Solve the linear system
    solver.Solve(&A,&u,&b);
	 
    // Save the solution
	 U.Save(t);
  }

  // Clean up
  delete equation;
}
//-----------------------------------------------------------------------------
