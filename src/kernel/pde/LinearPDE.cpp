// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006
//
// First added:  2004
// Last changed: 2006-05-07

#include <dolfin/dolfin_log.h>
#include <dolfin/FEM.h>
#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>
#include <dolfin/KrylovSolver.h>
#include <dolfin/LUSolver.h>
#include <dolfin/BilinearForm.h>
#include <dolfin/LinearForm.h>
#include <dolfin/Mesh.h>
#include <dolfin/BoundaryCondition.h>
#include <dolfin/Function.h>
#include <dolfin/LinearPDE.h>
#include <dolfin/Parametrized.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearPDE::LinearPDE(BilinearForm& a, LinearForm& L, Mesh& mesh)
  : GenericPDE(), _a(&a), _Lf(&L), _mesh(&mesh), _bc(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
LinearPDE::LinearPDE(BilinearForm& a, LinearForm& L, Mesh& mesh, 
  BoundaryCondition& bc) : GenericPDE(), _a(&a), _Lf(&L), _mesh(&mesh), _bc(&bc)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
LinearPDE::~LinearPDE()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint LinearPDE::solve(Function& u)
{
  dolfin_assert(_a);
  dolfin_assert(_Lf);
  dolfin_assert(_mesh);

  // Write a message
  dolfin_info("Solving static linear PDE.");

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
    solver.solve(*A, x, b);
  }
  else if ( solver_type == "iterative" || solver_type == "default" )
  {
    KrylovSolver solver(gmres);
    solver.set("parent", *this);
    solver.solve(*A, x, b);
  }
  else
    dolfin_error1("Unknown solver type \"%s\".", solver_type.c_str());

  delete A;

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint LinearPDE::elementdim()
{
  dolfin_assert(_a);
  return _a->trial().elementdim();
}
//-----------------------------------------------------------------------------
BilinearForm& LinearPDE::a()
{
  dolfin_assert(_a);
  return *_a;
}
//-----------------------------------------------------------------------------
LinearForm& LinearPDE::L()
{
  dolfin_assert(_Lf);
  return *_Lf;
}
//-----------------------------------------------------------------------------
Mesh& LinearPDE::mesh()
{
  dolfin_assert(_mesh);
  return *_mesh;
}
//-----------------------------------------------------------------------------
BoundaryCondition& LinearPDE::bc()
{
  dolfin_assert(_bc);
  return *_bc;
}
//-----------------------------------------------------------------------------
