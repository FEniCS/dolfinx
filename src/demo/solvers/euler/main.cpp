// Copyright (C) 2004 Harald Svensson.
// Licensed under the GNU GPL Version 2.
//
//     ---------------------------------------------------------------------------
//     Solver for 3D-Euler equations
//     ---------------------------------------------------------------------------
//
//     r    = density
//     v(i) = velocity : i=1,2,3
//     e    = energy
//
//     ---------------------------------------------------------------------------
//
//     p      = pressure = r*(kappa-1)*(e-v(i)*v(i)) : pressure
//     t(i,j) = ny*(dv(i)/dx(j) + dv(j)/dx(i))       : shear
//
//     ---------------------------------------------------------------------------
//
//     ny    = fluid dynamic viscosity
//     lamda = fluid thermal conductivity
//     kappa = fluid 
//
//     ---------------------------------------------------------------------------
//
//     Continuity : dr/dt    + div(rv) = 0
//     Momentum   : drv(i)/dt + v*grad(rv(i)) + dp/dx(i)- div am grad v(i) = fm(i) : i=1,2,3
//     Energy     : dre/dt    + v*grad(re)    - div ae grad e      = fe
//
//     ---------------------------------------------------------------------------
//

#include <dolfin.h>

using namespace dolfin;

// Momentum Source term
real fm(real x, real y, real z, real t, int i)
{

  real f = 0.0;

  switch ( i )
  {
    case 0:
      f = 0.0;
      break;
    case 1:
      f = 0.0;
      break;
    case 2:
      f = 0.0;
      break;
  }

  return f;

}

// Energy Source term
real fe(real x, real y, real z, real t)
{

  real f = 0.0;

  return f;

}

// Viscosity ( Momentum diffusivity )
real am(real x, real y, real z, real t)
{
  // Air at 0 degrees Celsius.
  real DynamicViscosity = 17.1E-3;

  real a = 0.0;

  a = DynamicViscosity;

  return a;

}

// Conductivity ( Energy diffusivity )
real ae(real x, real y, real z, real t)
{
  // Air at 0 degrees Celsius.
  real HeatConductivity = 18.9E-3;

  real a = 0.0;

  a = HeatConductivity;

  return a;

}

// Boundary conditions
void mybc(BoundaryCondition& bc)
{

  real X1inflow  =  0.0;
  real X2upper   =  1.0;
  real X2lower   = -1.0;
  //real X3left    =  1.0;
  //real X3right   = -1.0;
  real X1outflow =  1.0;

  if ( bc.coord().x ==  X1inflow )
  {
    // Inflow
    bc.set(BoundaryCondition::DIRICHLET,   1.20,  0);
    bc.set(BoundaryCondition::DIRICHLET,   0.1,   1);
    bc.set(BoundaryCondition::DIRICHLET,   0.0,   2);
    bc.set(BoundaryCondition::DIRICHLET, 273.0E3, 4);
  }
  else if ( bc.coord().y == X2lower )
  {
    // Lower wall
    //bc.set(BoundaryCondition::DIRICHLET, 1.2, 0);
    //bc.set(BoundaryCondition::NEUMANN,   0.0, 1);
    //bc.set(BoundaryCondition::DIRICHLET, 0.0, 2);
    //bc.set(BoundaryCondition::DIRICHLET, 0.0, 3);
  }
  else if ( bc.coord().y == X2upper )
  {
    // Upper wall
    //bc.set(BoundaryCondition::DIRICHLET, 1.2, 0);
    //bc.set(BoundaryCondition::NEUMANN,   0.0, 1);
    //bc.set(BoundaryCondition::DIRICHLET, 0.0, 2);
    //bc.set(BoundaryCondition::DIRICHLET, 0.0, 3);
  }
  else if ( bc.coord().x == X1outflow)
  {
    // Outflow
    //bc.set(BoundaryCondition::NEUMANN, 0.0, 0);
    //bc.set(BoundaryCondition::NEUMANN, 0.0, 1);
    //bc.set(BoundaryCondition::NEUMANN, 0.0, 2);
    //bc.set(BoundaryCondition::NEUMANN, 0.0, 3);
  }
  else
  {
    // Body
    //bc.set(BoundaryCondition::NEUMANN, 0.0);
    //bc.set(BoundaryCondition::DIRICHLET, 273.0E3, 1);
    //bc.set(BoundaryCondition::DIRICHLET, 0.0, 2);
    //bc.set(BoundaryCondition::DIRICHLET, 0.0, 3);
  }

}

int main(int argc, char **argv)
{

  Mesh mesh("dolfin.xml.gz");

  /*
  Problem euler("euler", mesh);

  euler.set("Source Momentum",fm);
  euler.set("Source Energy",fe);
  euler.set("Fluid Viscosity",am);
  euler.set("Fluid Conductivity",ae);

  euler.set("final time", 0.05);
  euler.set("time step", 0.01);

  euler.solve();
  */

  return 0;
}
