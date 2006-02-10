// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004
// Last changed: 2006-02-10

#include <dolfin/dolfin_log.h>
#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>
#include <dolfin/FEM.h>
#include <dolfin/GMRES.h>
#include <dolfin/LU.h>
#include <dolfin/BilinearForm.h>
#include <dolfin/LinearForm.h>
#include <dolfin/Mesh.h>
#include <dolfin/BoundaryCondition.h>
#include <dolfin/Function.h>
#include <dolfin/PDE.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
PDE::PDE(BilinearForm& a, LinearForm& L, Mesh& mesh)
  : Parametrized(), _a(&a), _L(&L), _mesh(&mesh), _bc(0)
{
  // Add parameters
  add("solver", "direct");
}
//-----------------------------------------------------------------------------
PDE::PDE(BilinearForm& a, LinearForm& L, Mesh& mesh, BoundaryCondition& bc)
  : Parametrized(), _a(&a), _L(&L), _mesh(&mesh), _bc(&bc)
{
  // Add parameters
  add("solver", "direct");
}
//-----------------------------------------------------------------------------
PDE::~PDE()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Function PDE::solve()
{
  Function u;
  solve(u);
  return u;
}
//-----------------------------------------------------------------------------
void PDE::solve(Function& u)
{
  dolfin_assert(_a);
  dolfin_assert(_L);
  dolfin_assert(_mesh);

  dolfin_info("Solving static linear PDE.");

  // Make sure u is a discrete function associated with the trial space
  u.init(*_mesh, _a->trial());
  Vector& x = u.vector();

  // Assemble linear system
  Matrix A;
  Vector b;
  if ( _bc )
    FEM::assemble(*_a, *_L, A, b, *_mesh, *_bc);
  else
    FEM::assemble(*_a, *_L, A, b, *_mesh);
  
  // Solve the linear system
  const std::string solver_type = get("solver");
  if ( solver_type == "direct" )
  {
    LU solver;
    solver.solve(A, x, b);
  }
  else if ( solver_type == "iterative" )
  {
    GMRES solver;
    solver.solve(A, x, b);
  }
  else
    dolfin_error1("Unknown solver type \"%s\".", solver_type.c_str());
}
//-----------------------------------------------------------------------------
BilinearForm& PDE::a()
{
  dolfin_assert(_a);
  return *_a;
}
//-----------------------------------------------------------------------------
LinearForm& PDE::L()
{
  dolfin_assert(_L);
  return *_L;
}
//-----------------------------------------------------------------------------
Mesh& PDE::mesh()
{
  dolfin_assert(_mesh);
  return *_mesh;
}
//-----------------------------------------------------------------------------
