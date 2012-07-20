// Copyright (C) 2007-2009 Anders Logg
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
// Modified by Garth N. Wells, 2007-2008.
// Modified by Ola Skavhaug, 2008.
//
// First added:  2007-01-17
// Last changed: 2009-10-06

#ifndef __ASSEMBLER_TOOLS_H
#define __ASSEMBLER_TOOLS_H

#include <string>
#include <utility>
#include <vector>
#include <dolfin/common/types.h>

namespace dolfin
{

  // Forward declarations
  class GenericTensor;
  class Form;

  /// This class provides some common functions used in the
  /// Assembler and SystemAssembler classes.

  class AssemblerTools
  {
  public:

    // Check form
    static void check(const Form& a);

    // Initialize global tensor
    static void init_global_tensor(GenericTensor& A, const Form& a,
         const std::vector<std::pair<std::pair<uint, uint>, std::pair<uint, uint> > >& periodic_master_slave_dofs,
         bool reset_sparsity, bool add_values);

    // Pretty-printing for progress bar
    static std::string progress_message(uint rank, std::string integral_type);

  };

}

#endif
