// Copyright (C) 2013 Anders Logg
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
// First added:  2013-09-12
// Last changed: 2013-09-18

#include <dolfin/log/log.h>
#include "CCFEMAssembler.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void CCFEMAssembler::assemble(GenericTensor& A, const CCFEMForm& a)
{
  begin(PROGRESS, "Assembling tensor over CCFEM function space.");

  // Initialize global tensor
  init_global_tensor(A, a);

  end();
}
//-----------------------------------------------------------------------------
void CCFEMAssembler::init_global_tensor(GenericTensor& A, const CCFEMForm& a)
{

}
//-----------------------------------------------------------------------------
