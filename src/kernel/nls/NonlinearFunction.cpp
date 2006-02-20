// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-24
// Last changed: 2005-12-05

#include <dolfin/FEM.h>
#include <dolfin/NonlinearPDE.h>
#include <dolfin/BilinearForm.h>
#include <dolfin/LinearForm.h>

using namespace dolfin;
NonlinearFunction::NonlinearFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NonlinearFunction::~NonlinearFunction()
{
  // Do nothing 
}
//-----------------------------------------------------------------------------
void NonlinearFunction::form(Matrix& A, Vector& b, const Vector& x)
{
  dolfin_error("Nonlinear function update for F(u) and J  has not been supplied by user.");
}
//-----------------------------------------------------------------------------
void NonlinearFunction::F(Vector& b, const Vector& x)
{
  dolfin_error("Nonlinear function update for F(u)  has not been supplied by user.");
}
//-----------------------------------------------------------------------------
void NonlinearFunction::J(Matrix& A, const Vector& x)
{
  dolfin_error("Nonlinear function update for Jacobian has not been supplied by user.");
}
//-----------------------------------------------------------------------------
