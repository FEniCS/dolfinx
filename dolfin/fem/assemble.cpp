// Copyright (C) 2007-2015 Anders Logg
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

#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/Scalar.h>
#include <dolfin/mesh/Mesh.h>
#include "Form.h"
#include "MultiMeshForm.h"
#include "Assembler.h"
#include "SystemAssembler.h"
#include "MultiMeshAssembler.h"
#include "assemble.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A, const Form& a)
{
  Assembler assembler;
  assembler.assemble(A, a);
}
//-----------------------------------------------------------------------------
void dolfin::assemble_system(GenericMatrix& A, GenericVector& b,
                             const Form& a, const Form& L,
                             std::vector<std::shared_ptr<const DirichletBC>> bcs)
{
  SystemAssembler assembler(reference_to_no_delete_pointer(a),
                            reference_to_no_delete_pointer(L), bcs);
  assembler.assemble(A, b);
}
//-----------------------------------------------------------------------------
void dolfin::assemble_system(GenericMatrix& A, GenericVector& b,
                             const Form& a, const Form& L,
                             std::vector<std::shared_ptr<const DirichletBC>> bcs,
                             const GenericVector& x0)
{
  SystemAssembler assembler(reference_to_no_delete_pointer(a),
                            reference_to_no_delete_pointer(L), bcs);
  assembler.assemble(A, b, x0);
}
//-----------------------------------------------------------------------------
void dolfin::assemble_multimesh(GenericTensor& A, const MultiMeshForm& a)
{
  MultiMeshAssembler assembler;
  assembler.assemble(A, a);
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

  // Create Scalar with appropriate communicator
  dolfin_assert(a.mesh());
  Scalar s(a.mesh()->mpi_comm());

  // Assemble
  Assembler assembler;
  assembler.assemble(s, a);
  return s.get_scalar_value();
}
//-----------------------------------------------------------------------------
double dolfin::assemble_multimesh(const MultiMeshForm& a)
{
  if (a.rank() != 0)
  {
    dolfin_error("assemble.cpp",
                 "assemble MultiMeshForm",
                 "Expecting a scalar form but rank is %d",
                 a.rank());
  }

  Scalar s;
  MultiMeshAssembler assembler;
  assembler.assemble(s, a);
  return s.get_scalar_value();
}
//-----------------------------------------------------------------------------
