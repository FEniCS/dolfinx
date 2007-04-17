// Copyright (C) 2004-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006
//
// First added:  2004
// Last changed: 2007-04-17

#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>
#include <dolfin/BoundaryCondition.h>
#include <dolfin/Assembler.h>
#include <dolfin/LUSolver.h>
#include <dolfin/KrylovSolver.h>
#include <dolfin/LinearPDE.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearPDE::LinearPDE(Form& a,
                     Form& L,
                     Mesh& mesh,
                     Array<BoundaryCondition*> bcs)
  : GenericPDE(a, L, mesh, bcs)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
LinearPDE::~LinearPDE()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void LinearPDE::solve(Function& u)
{
  dolfin_begin("Solving static linear PDE.");
    
  // Assemble linear system
  Matrix A;
  Vector b;
  Assembler assembler;
  assembler.assemble(A, a, mesh);
  assembler.assemble(b, L, mesh);

  // Apply boundary conditions
  for (uint i = 0; i < bcs.size(); i++)
    bcs[i]->apply(A, b, a);

  // Create solution vector
  Vector* x = new Vector();

  // Solve linear system
  const std::string solver_type = get("PDE linear solver");
  if ( solver_type == "direct" )
  {
    LUSolver solver;
    solver.set("parent", *this);
    solver.solve(A, *x, b);
  }
  else if ( solver_type == "iterative" || solver_type == "default" )
  {
    KrylovSolver solver(gmres);
    solver.set("parent", *this);
    solver.solve(A, *x, b);
  }
  else
    dolfin_error1("Unknown solver type \"%s\".", solver_type.c_str());

  x->disp();

  delete x;

  dolfin_end();
}
//-----------------------------------------------------------------------------
