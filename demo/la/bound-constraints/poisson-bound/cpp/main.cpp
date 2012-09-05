// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2006-02-07
// Last changed: 2011-07-01
//
// This demo program solves Poisson's equation
//
//     - div grad u(x, y) = f(x, y)
//
// on the unit square with source f given by
//
//     f(x, y) = 10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)
//
// and boundary conditions given by
//
//     u(x, y) = 0        for x = 0 or x = 1
// du/dn(x, y) = sin(5*x) for y = 0 or y = 1

#include <dolfin.h>
#include "Poisson.h"
#include "taosolver.h"
#include "petscdm.h"
#include "petscksp.h"
#include "petscvec.h" 
#include "petscmat.h"
//#include "TAOLinearBoundSolver.h"

static  char help[]=
"This example  solve with TAO a Poisson equation assembled with FEniCS";

using namespace dolfin;


using namespace dolfin;

// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    double dx = x[0] - 0.5;
    double dy = x[1] - 0.5;
    values[0] = 10*exp(-(dx*dx + dy*dy) / 0.02);
  }
};

// Normal derivative (Neumann boundary condition)
class dUdN : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(5*x[0]);
  }
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS;
  }
};


class LowerBound : public Expression
{
public:

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = 0.;
  }

};


class UpperBound : public Expression
{
public:

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = .2*sin(x[0]);
  }

};

/* User-defined routines */
//static PetscErrorCode Monitor(TaoSolver, void*);
//static PetscErrorCode ConvergenceTest(TaoSolver, void*);

int main(int argc, char **argv)
{ 
  // Create mesh and function space
  UnitSquare mesh(32, 32);
  Poisson::FunctionSpace V(mesh);

  // Define boundary condition
  Constant u0(0.0);
  DirichletBoundary boundary;
  DirichletBC bc(V, u0, boundary);

  // Define variational forms
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  Source f;
  dUdN g;
  L.f = f;
  L.g = g;

  // Compute solution
  Function u(V);
  Function usol(V); // Create a function
  //Function Ulower(V); // Create a function
  //Function Uupper(V); // Create a function

  PETScVector& x = (*usol.vector()).down_cast<PETScVector>(); //Create the PetscVector associated to the function
  PETScMatrix A;
  PETScVector b;//ierr = VecCopy(x,*UP.vec()); CHKERRQ(ierr);
  
  assemble(A, a);
  assemble(b, L);
  bc.apply(A);
  bc.apply(b);
  
  //PETScVector xl=x;
  //PETScVector xu=x;
  
  // Interpolate expression into V0
  UpperBound xu_exp;
  Function xu_f(V);
  xu_f.interpolate(xu_exp);
  PETScVector& xu = (*xu_f.vector()).down_cast<PETScVector>(); //Create the PetscVector associated to the function
  //
  LowerBound xl_exp;
  Function xl_f(V);
  xl_f.interpolate(xl_exp);  
  PETScVector& xl = (*xl_f.vector()).down_cast<PETScVector>(); //Create the PetscVector associated to the function
  
  
  /* The TAO code begins here */
  TAOLinearBoundSolver TAOSolver;
  TAOSolver.solve(A, x, b, xl, xu);
  //plot(usol);
  return 0;
}

