// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

dolfin_bc my_bc(real x, real y, real z, int node, int component);

real f   (real x, real y, real z, real t);
real eps (real x, real y, real z, real t);
real bx  (real x, real y, real z, real t);
real by  (real x, real y, real z, real t);

int main(int argc, char **argv)
{
  // Set problem
  dolfin_set_problem("conv-diff");
  
   // Set parameters
  dolfin_set_parameter("problem description", "Convection-Diffusion around a DOLFIN");
  dolfin_set_parameter("grid file",           "../../../data/grids/dolfin-1.inp");
  dolfin_set_parameter("output file prefix",  "conv_diff");
  dolfin_set_parameter("output file type",    "gid");
  dolfin_set_parameter("start time",          0.0);
  dolfin_set_parameter("final time",          0.25);
  dolfin_set_parameter("space dimension",     2);
  
  // Set boundary conditions
  dolfin_set_boundary_conditions(my_bc);

  // Set coefficients and right-hand side
  dolfin_set_function("source",f);
  dolfin_set_function("diffusivity",eps);
  dolfin_set_function("x-convection",bx);
  dolfin_set_function("y-convection",by);
  
  // Initialize dolfin
  dolfin_init(argc,argv);

  // Save parameters
  dolfin_save_parameters();

  // Solve the problem
  dolfin_solve();
  
  // Finished
  dolfin_end();
  
  return 0;
}

dolfin_bc my_bc(real x, real y, real z, int node, int component)
{
  dolfin_bc bc;

  // u = 0 on the inflow boundary
  if ( x == 1.0 ){
	 bc.type = dirichlet;
	 bc.val  = 0.0;
  }
  
  // u = 1 on the dolphin
  if ( (node < 77) || ( (node >= 759) & (node <= 883) ) ){
  	 bc.type = dirichlet;
  	 bc.val  = 1.0;
  }

  return bc;
}

real f(real x, real y, real z, real t)
{
  return 0.0;
}

real eps (real x, real y, real z, real t)
{
  return 0.1;
}

real bx(real x, real y, real z, real t)
{
  return -10.0;
}

real by(real x, real y, real z, real t)
{
  return 0.0;
}
