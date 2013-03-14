// Copyright (C) 2012 Corrado Maurini
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
// Modified by Corrado Maurini 2013
// 
// First added:  2012-09-03
// Last changed: 2013-03-13
//
// This demo program uses of the interface to TAO solver for variational inequalities 
// to solve a contact mechanics problems in FEnics. 
// The example considers a heavy elastic circle in a box of the same diameter

#include <dolfin.h>
#include "Elasticity.h"

using namespace dolfin;

int main()
{
  #ifdef HAS_TAO
    
    // Lower bound for displacement
    class LowerBound : public Expression
    {
    public:
    
      LowerBound() : Expression(2) {}
    
      void eval(Array<double>& values, const Array<double>& x) const
      {
        double xmin = -1.-DOLFIN_EPS;
        double ymin = -1.;
        values[0] = xmin-x[0];
        values[1] = ymin-x[1];
      }
    
    };
    
    // Upper bound for displacement
    class UpperBound : public Expression
    {
    public:
    
      UpperBound() : Expression(2) {}

      void eval(Array<double>& values, const Array<double>& x) const
      {
        double xmax = 1.+DOLFIN_EPS;
        double ymax = 1.;
        values[0] = xmax-x[0];
        values[1] = ymax-x[1];
      }
    
    };

  // Read mesh and create function space
  UnitCircleMesh mesh(30);
  Elasticity::FunctionSpace V(mesh);

  // Create right-hand side
  Constant f(0.0, -0.4);

  // Set elasticity parameters
  double E  = 10.0;
  double nu = 0.3;
  Constant mu(E / (2*(1 + nu)));
  Constant lambda(E*nu / ((1 + nu)*(1 - 2*nu)));

  // Define variational problem
  Elasticity::BilinearForm a(V, V);
  a.mu = mu; a.lmbda = lambda;
  Elasticity::LinearForm L(V);
  L.f = f;
  Function usol(V);
  Function Ulower(V); // Create a function
  Function Uupper(V); // Create a function
  
  // Assemble the matrix and vector for linear elasticity
  PETScMatrix A;
  PETScVector b;
  assemble(A, a);
  assemble(b, L);
    
  // Interpolate expression for Upper bound 
  UpperBound xu_exp;
  Function xu_f(V);
  xu_f.interpolate(xu_exp);
 
  // Interpolate expression for Upper bound 
  LowerBound xl_exp;
  Function xl_f(V);
  xl_f.interpolate(xl_exp);  
    
  // Create the PetscVector associated to the functions
  PETScVector& x  = (*usol.vector()).down_cast<PETScVector>(); // Solution
  PETScVector& xl = (*xl_f.vector()).down_cast<PETScVector>(); // Lower bound
  PETScVector& xu = (*xu_f.vector()).down_cast<PETScVector>(); // Upper bound

  // Solve the problem with the TAO Solver 
  TAOLinearBoundSolver TAOSolver("tao_tron","gmres");
  TAOSolver.solve(A, x, b, xl, xu);
  
  // Plot solution
  plot(usol, "Displacement", "displacement");
  
  // Make plot windows interactive
  interactive();
  #endif
  
 return 0;
}
