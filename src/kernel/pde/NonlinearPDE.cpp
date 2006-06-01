// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2005-10-24
// Last changed: 2006-05-07

#ifdef HAVE_PETSC_H

#include <dolfin/FEM.h>
#include <dolfin/NonlinearPDE.h>
#include <dolfin/BilinearForm.h>
#include <dolfin/LinearForm.h>

using namespace dolfin;

NonlinearPDE::NonlinearPDE(BilinearForm& a, LinearForm& L, Mesh& mesh) : 
      GenericPDE(), _a(&a), _Lf(&L), _mesh(&mesh), _bc(0)
{
  newton_solver.set("parent", *this);
}
//-----------------------------------------------------------------------------
NonlinearPDE::NonlinearPDE(BilinearForm& a, LinearForm& L, Mesh& mesh, 
      BoundaryCondition& bc) : GenericPDE(), _a(&a), _Lf(&L), 
      _mesh(&mesh), _bc(&bc)
{
  newton_solver.set("parent", *this);
}
//-----------------------------------------------------------------------------
NonlinearPDE::~NonlinearPDE()
{
  // Do nothing 
}
//-----------------------------------------------------------------------------
void NonlinearPDE::form(Matrix& A, Vector& b, const Vector& x)
{
  if(!_a)
  {  
    dolfin_error("Nonlinear function update for F(u) and Jacobian has not been supplied by user.");
  }  
  else
  {
    FEM::assemble(*_a, *_Lf, A, b, *_mesh);
    if(_bc) 
    { 
      FEM::applyBC(A, *_mesh, _a->test(), *_bc);
      FEM::assembleResidualBC(b, x, *_mesh, _Lf->test(), *_bc);
    }
    else
    {
      //FIXME: Deal with zero Neumann condition on entire boundary here. Need to
      //fix FEM::assembleResidualFEM::assembleResidualBC(b, x, *_mesh, _Lf->test());
      dolfin_error("Pure zero Neumann boundary conditions not yet implemented for nonlinear PDE.");
    }
  }
  
}
//-----------------------------------------------------------------------------
//void NonlinearPDE::F(Vector& b, const Vector& x)
//{
//  dolfin_error("Nonlinear PDE update for F(u)  has not been supplied by
//user.");
//}
//-----------------------------------------------------------------------------
//void NonlinearPDE::J(Matrix& A, const Vector& x)
//{
//  dolfin_error("Nonlinear PDE update for Jacobian has not been supplied by
//user.");
//}
//-----------------------------------------------------------------------------
dolfin::uint NonlinearPDE::solve(Function& u)
{
  // Initialise function if necessary
  if (u.type() != Function::discrete)
    u.init(*_mesh, _a->trial());

  Vector& x = u.vector();  

  // Solve nonlinear problem using u as start value
  return newton_solver.solve(*this, x);
}
//-----------------------------------------------------------------------------
dolfin::uint NonlinearPDE::elementdim()
{
  dolfin_assert(_a);
  return _a->trial().elementdim();
}
//-----------------------------------------------------------------------------
BilinearForm& NonlinearPDE::a()
{
  dolfin_assert(_a);
  return *_a;
}
//-----------------------------------------------------------------------------
LinearForm& NonlinearPDE::L()
{
  dolfin_assert(_Lf);
  return *_Lf;
}
//-----------------------------------------------------------------------------
Mesh& NonlinearPDE::mesh()
{
  dolfin_assert(_mesh);
  return *_mesh;
}
//-----------------------------------------------------------------------------
BoundaryCondition& NonlinearPDE::bc()
{
  dolfin_assert(_bc);
  return *_bc;
}
//-----------------------------------------------------------------------------

#endif
