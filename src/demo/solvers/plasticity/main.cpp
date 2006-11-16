// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#include <dolfin.h>
#include <dolfin/VonMises.h>

using namespace dolfin;

// Right-hand side (body force)
class MyFunction : public Function
{
  real eval(const Point& p, unsigned int i)
  {
    if (i==1)
      return 0.0*time(); 
    else
      return 0.0;
  }
};

// Boundary conditions
class MyBC : public BoundaryCondition
{
  void eval(BoundaryValue& value, const Point& p, unsigned int i)
  {
    if ( (std::abs(p.y()) < DOLFIN_EPS && i ==1) || ( (std::abs(p.y()) < DOLFIN_EPS) && ( (p.x()-0.24) > DOLFIN_EPS) && ( (p.x()-0.76) < DOLFIN_EPS)  && i==0) )
      value = 0.0;

     if (std::abs(p.y() - 1.0) < DOLFIN_EPS && i==1)
      value = -0.002*time();
  }
};

int main()
{
  UnitSquare mesh(20, 20);

  MyFunction f;
  MyBC bc;

  // Young's modulus and Poisson's ratio
  real E = 200000.0;
  real nu = 0.3;

  // Final time and time step
  real T = 1.0;
  real dt = 0.1;

  // Slope of hardening (linear) and hardening parameter
  real E_t(0.1 * E);
  real hardening_parameter = E_t/(1-E_t/E);

  // Yield stress
  real yield_stress = 200.0;

  // Object of class von Mise
  VonMises J2(E, nu, yield_stress, hardening_parameter);

  // Solve problem
  PlasticitySolver::solve(mesh, bc, f, dt, T, J2);

  return 0;
}
