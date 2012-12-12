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

#ifndef __ASSEMBLER_BASE_H
#define __ASSEMBLER_BASE_H

#include <string>
#include <utility>
#include <vector>
#include <dolfin/common/types.h>

namespace dolfin
{

  // Forward declarations
  class GenericTensor;
  class Form;

  /// This class provides some common functions used in
  /// assembler classes.

  class AssemblerBase
  {
  public:

    // Check form
    AssemblerBase() : reset_sparsity(true), add_values(false),
        finalize_tensor(true), keep_diagonal(false) {}

    /// reset_sparsity (bool)
    ///     Default value is true.
    ///     This controls whether the sparsity pattern of the
    ///     given tensor is reset prior to assembly.
    bool reset_sparsity;

    /// add_values (bool)
    ///     Default value is false.
    ///     This controls whether values are added to the given
    ///     tensor or if it is zeroed prior to assembly.
    bool add_values;

    /// finalize_tensor (bool)
    ///     Default value is true.
    ///     This controls whether the assembler finalizes the
    ///     given tensor after assembly is completed by calling
    ///     A.apply().
    bool finalize_tensor;

    /// keep_diagonal (bool)
    ///     Default value is true.
    ///     This controls whether the assembler enures that a diagonal
    ///     entry exists in an assembled matrix. It may be removed
    ///     if the matrix is finalised.
    bool keep_diagonal;

    // Initialize global tensor
    void init_global_tensor(GenericTensor& A, const Form& a,
         const std::vector<std::pair<std::pair<std::size_t, std::size_t>, std::pair<std::size_t, std::size_t> > >& periodic_master_slave_dofs);

  protected:

    // Check for compatibility of reset_sparsity, add_values,
    // finalize_tensor and keep_diagonal
    void check_parameters() const;

    // Check form
    static void check(const Form& a);

    // Pretty-printing for progress bar
    static std::string progress_message(std::size_t rank, std::string integral_type);

  };

}

#endif
