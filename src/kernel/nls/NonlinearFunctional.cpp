// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-24
// Last changed: 2005

#include <dolfin/NonlinearFunctional.h>

using namespace dolfin;
NonlinearFunctional::NonlinearFunctional() : _a(0), _L(0), _mesh(0), _x0(0), 
   _A(0), _b(0), _bc(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NonlinearFunctional::NonlinearFunctional(BilinearForm& a, LinearForm& L, 
  Mesh& mesh, Vector& x0, Matrix& A, Vector& b, BoundaryCondition& bc) : _a(&a),
   _L(&L), _mesh(&mesh), _x0(&x0), _A(&A), _b(&b), _bc(&bc)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NonlinearFunctional::~NonlinearFunctional()
{
  // Do nothing 
}
//-----------------------------------------------------------------------------
void NonlinearFunctional::UpdateNonlinearFunction()
{
//  cout << "Inside UpdateNonlinearFunction " << endl;
  dolfin_warning("Nonlinear function update has not been supplied by user. Nothing updated");
}
//-----------------------------------------------------------------------------
Mesh& NonlinearFunctional::rmesh()
{
  return *_mesh;
}
//-----------------------------------------------------------------------------
