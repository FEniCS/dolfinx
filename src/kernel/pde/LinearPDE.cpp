// Copyright (C) 2004-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006
//
// First added:  2004
// Last changed: 2007-04-17

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
  dolfin_error("Not implemented.");

  dolfin_info("Solving static linear PDE.");
    
  /*

    
    // Make sure u is a discrete function associated with the trial space
    u.init(*_mesh, _a->trial());
    Vector& x = u.vector();
    
    // Get solver type
  const std::string solver_type = get("PDE linear solver");

  // Assemble linear system
  Vector b;
  Matrix* A;
  if ( solver_type == "direct" )
#ifdef HAVE_PETSC_H
    A = new Matrix(Matrix::umfpack);
#else
    A = new Matrix;
#endif
  else
    A = new Matrix;

  if ( _bc )
    FEM::assemble(*_a, *_Lf, *A, b, *_mesh, *_bc);
  else
    FEM::assemble(*_a, *_Lf, *A, b, *_mesh);

  // Solve the linear system
  if ( solver_type == "direct" )
  {
    LUSolver solver;
    solver.set("parent", *this);
    solver.solve(A, x, b);
  }
  else if ( solver_type == "iterative" || solver_type == "default" )
  {
    KrylovSolver solver(gmres);
    solver.set("parent", *this);
    solver.solve(A, x, b);
  }
  else
    dolfin_error1("Unknown solver type \"%s\".", solver_type.c_str());

  delete A;

*/
}
//-----------------------------------------------------------------------------
