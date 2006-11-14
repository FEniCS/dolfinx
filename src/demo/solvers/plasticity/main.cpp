// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#include <dolfin.h>
#include <dolfin/VonMises.h>

using namespace dolfin;

// Right-hand side
class MyFunction : public Function //this function controls the body force
{
  real eval(const Point& p, unsigned int i)
  {
    if (i==1)  // bodyforce in y-dir
      return 0.0*time();  // 

    else
      return 0.0;  // remaining directions
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

  real E = 200000.0; // Young's modulus
  real nu = 0.3; // Poisson's ratio
  
  real T = 1.0;  // final time
  real dt = 0.1; // time step

  // hardening
  real E_t(0.1 * E); // slope of hardening (linear)
  real H = E_t/(1-E_t/E); // hardening parameter (linear)

  // yield stress
  real sig_o = 200.0;

  // object of class von Mise
  VonMises J2(sig_o, H);

  // solve problem
  PlasticitySolver::solve(mesh, bc, f, E, nu, dt, T, J2);

  return 0;
}
