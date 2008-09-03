// Copyright (C) 2006-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-10-24
// Last changed: 2008-08-26

#include <dolfin/log/dolfin_log.h>
#include "NonlinearProblem.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
NonlinearProblem::NonlinearProblem()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NonlinearProblem::~NonlinearProblem()
{
  // Do nothing 
}
//-----------------------------------------------------------------------------
void NonlinearProblem::form(GenericMatrix& A, GenericVector& b, const GenericVector& x)
{
  error("The function NonlinearProblem::form will be removed. Supply the functions NonlinearProblem::F and NonlinearProblem::J .");
}
//-----------------------------------------------------------------------------
void NonlinearProblem::F(GenericVector& b, const GenericVector& x)
{
  error("Nonlinear problem update for F(u) has not been supplied by user.");
}
//-----------------------------------------------------------------------------
void NonlinearProblem::J(GenericMatrix& A, const GenericVector& x)
{
  error("Nonlinear problem update for Jacobian has not been supplied by user.");
}
//-----------------------------------------------------------------------------
