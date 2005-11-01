// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-24
// Last changed: 2005

#include <dolfin/NonlinearFunction.h>

using namespace dolfin;
NonlinearFunction::NonlinearFunction() : _a(0), _L(0), _mesh(0), _bc(0),
                                         _A(0), _b(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NonlinearFunction::NonlinearFunction(BilinearForm& a, LinearForm& L, Mesh& mesh,
  BoundaryCondition& bc) : _a(&a), _L(&L), _mesh(&mesh), _bc(&bc), _A(0), _b(0) 
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NonlinearFunction::~NonlinearFunction()
{
  // Do nothing 
}
//-----------------------------------------------------------------------------
void NonlinearFunction::update(Vector& x)
{
//  cout << "Inside UpdateNonlinearFunction " << endl;
  dolfin_warning("Nonlinear function update has not been supplied by user. Nothing updated");
}
//-----------------------------------------------------------------------------
BilinearForm& NonlinearFunction::a()
{
  if( !_a)
    dolfin_error("Bilinear form has not been specified.");

  return *_a;
}
//-----------------------------------------------------------------------------
LinearForm& NonlinearFunction::L()
{
  if( !_L)
    dolfin_error("Linear form has not been specified.");

  return *_L;
}
//-----------------------------------------------------------------------------
Mesh& NonlinearFunction::mesh()
{
  if( !_mesh)
    dolfin_error("Mesh has not been specified.");

  return *_mesh;
}
//-----------------------------------------------------------------------------
BoundaryCondition& NonlinearFunction::bc()
{
  if( !_bc)
    dolfin_error("Mesh has not been specified.");

  return *_bc;
}
//-----------------------------------------------------------------------------
