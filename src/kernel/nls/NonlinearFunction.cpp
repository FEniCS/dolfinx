// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-24
// Last changed: 2005

#include <dolfin/NonlinearFunction.h>

using namespace dolfin;
NonlinearFunction::NonlinearFunction() : _A(0), _b(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NonlinearFunction::~NonlinearFunction()
{
  // Do nothing 
}
//-----------------------------------------------------------------------------
void NonlinearFunction::setF(Vector& b) 
{
  _b = &b; 
}
//-----------------------------------------------------------------------------
void NonlinearFunction::setJ(Matrix& A) 
{
  _A = &A; 
}
//-----------------------------------------------------------------------------
dolfin::uint NonlinearFunction::size()
{
  dolfin_error("Nonlinear function size has not been supplied by user.");
  return 1;
}
//-----------------------------------------------------------------------------
dolfin::uint NonlinearFunction::nzsize()
{
  dolfin_error("Maximum number of nonzeros per rwo (nzsize) has not been supplied by user.");
  return 1;
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
void NonlinearFunction::form(Matrix& A, Vector& b, const Vector& x)
{
  dolfin_error("Nonlinear function update for F(u) and Jacobian has not been supplied by user.");
}
//-----------------------------------------------------------------------------
Matrix& NonlinearFunction::J() const
{
  if( !_A )
    dolfin_error("Jacobian matrix has not been specified.");

  return *_A;
}
//-----------------------------------------------------------------------------
Vector& NonlinearFunction::F() const
{
  if( !_b )
    dolfin_error("RHS vector has not been specified.");

  return *_b;
}
//-----------------------------------------------------------------------------
