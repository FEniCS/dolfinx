// Copyright (C) 2007-2013 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2008-2013.
// Modified by Johan Hake, 2009.
// Modified by Joachim B. Haga, 2012.
// Modified by Martin S. Alnaes, 2013.
//
// First added:  2007-01-17
// Last changed: 2013-02-13

#include <dolfin/la/Scalar.h>
#include "Form.h"
#include "Assembler.h"
#include "SystemAssembler.h"
#include "assemble.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A, const Form& a)
{
  Assembler assembler;
  assembler.assemble(A, a);
}
//-----------------------------------------------------------------------------
void dolfin::assemble_system(GenericMatrix& A,
                             GenericVector& b,
                             const Form& a,
                             const Form& L)
{
  SystemAssembler assembler(a, L);
  assembler.assemble(A, b);
}
//-----------------------------------------------------------------------------
void dolfin::assemble_system(GenericMatrix& A,
                             GenericVector& b,
                             const Form& a,
                             const Form& L,
                             const DirichletBC& bc)
{
  SystemAssembler assembler(a, L, bc);
  assembler.assemble(A, b);
}
//-----------------------------------------------------------------------------
void dolfin::assemble_system(GenericMatrix& A,
                             GenericVector& b,
                             const Form& a,
                             const Form& L,
                             const std::vector<const DirichletBC*> bcs)
{
  SystemAssembler assembler(a, L, bcs);
  assembler.assemble(A, b);
}
//-----------------------------------------------------------------------------
void dolfin::assemble_system(GenericMatrix& A,
                             GenericVector& b,
                             const Form& a,
                             const Form& L,
                             const std::vector<const DirichletBC*> bcs,
                             const GenericVector* x0)
{
  SystemAssembler assembler(a, L, bcs);
  assembler.assemble(A, b, x0);
}
//-----------------------------------------------------------------------------
double dolfin::assemble(const Form& a)
{
  if (a.rank() != 0)
  {
    dolfin_error("assemble.cpp",
                 "assemble form",
                 "Expecting a scalar form but rank is %d",
                 a.rank());
  }

  Scalar s;
  Assembler assembler;
  assembler.assemble(s, a);
  return s;
}
//-----------------------------------------------------------------------------
