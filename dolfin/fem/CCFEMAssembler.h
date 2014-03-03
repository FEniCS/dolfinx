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
// Last changed: 2014-03-03

#ifndef __CCFEM_ASSEMBLER_H
#define __CCFEM_ASSEMBLER_H

#include "AssemblerBase.h"
#include "Assembler.h"

namespace dolfin
{

  // Forward declarations
  class GenericTensor;
  class CCFEMForm;

  /// This class implements functionality for finite element assembly
  /// over cut and composite finite element (CCFEM) function spaces.

  class CCFEMAssembler : public AssemblerBase
  {
  public:

    /// Constructor
    CCFEMAssembler();

    /// Assemble tensor from given form
    ///
    /// *Arguments*
    ///     A (_GenericTensor_)
    ///         The tensor to assemble.
    ///     a (_Form_)
    ///         The form to assemble the tensor from.
    void assemble(GenericTensor& A, const CCFEMForm& a);

  private:

    // Assemble over cells
    void assemble_cells(GenericTensor& A, const CCFEMForm& a);

    // Initialize global tensor
    void init_global_tensor(GenericTensor& A, const CCFEMForm& a);

  };

}

#endif
