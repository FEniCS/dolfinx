// Copyright (C) 2004-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006
//
// First added:  2004
// Last changed: 2007-05-13

#include <dolfin/Matrix.h>
#include <dolfin/BoundaryCondition.h>
#include <dolfin/Assembler.h>
#include <dolfin/LUSolver.h>
#include <dolfin/KrylovSolver.h>
#include <dolfin/Function.h>
#include <dolfin/LinearPDE.h>
#include <dolfin/dolfin_io.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearPDE::LinearPDE(Form& a,
                     Form& L,
                     Mesh& mesh,
                     Array<BoundaryCondition*>& bcs)
  : GenericPDE(a, L, mesh, bcs)
{
  dolfin_info("Creating linear PDE with %d boundary condition(s).", bcs.size());
}
//-----------------------------------------------------------------------------
LinearPDE::~LinearPDE()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void LinearPDE::solve(Function& u)
{
  dolfin_begin("Solving linear PDE.");
    
  // Assemble linear system
  Matrix A;
  Vector b;
  Assembler assembler;
  assembler.assemble(A, a, mesh);
  assembler.assemble(b, L, mesh);

  // Apply boundary conditions
  for (uint i = 0; i < bcs.size(); i++)
    bcs[i]->apply(A, b, a);

  // Solve linear system
  const std::string solver_type = get("PDE linear solver");
  if ( solver_type == "direct" )
  {
    cout << "Using direct solver." << endl;
    LUSolver solver;
    solver.set("parent", *this);
    solver.solve(A, x, b);
  }
  else if ( solver_type == "iterative" )
  {
    cout << "Using iterative solver (GMRES)." << endl;
    KrylovSolver solver(gmres);
    solver.set("parent", *this);
    solver.solve(A, x, b);
  }
  else
    dolfin_error1("Unknown solver type \"%s\".", solver_type.c_str());

  //cout << "Matrix:" << endl;
  //A.disp();

  //cout << "Vector:" << endl;
  //b.disp();

  //cout << "Solution vector:" << endl;
  //x.disp();

  // Set function data
  u.init(mesh, x, a, 1);

  dolfin_end();
}
//-----------------------------------------------------------------------------
