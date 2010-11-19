// Copyright (C) 2007-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007-2008.
// Modified by Ola Skavhaug, 2008.
//
// First added:  2007-01-17
// Last changed: 2009-10-06

#ifndef __ASSEMBLER_TOOLS_H
#define __ASSEMBLER_TOOLS_H

#include <vector>
#include <dolfin/common/types.h>

namespace dolfin
{

  // Forward declarations
  class GenericTensor;
  class Form;
  class UFC;

  /// This class provides some common functions used in the
  /// Assembler and SystemAssembler classes.

  class AssemblerTools
  {
  public:

    // Check form
    static void check(const Form& a);

    // Initialize global tensor
    static void init_global_tensor(GenericTensor& A,
                                   const Form& a,
                                   bool reset_sparsity,
                                   bool add_values);

    // Pretty-printing for progress bar
    static std::string progress_message(uint rank,
                                        std::string integral_type);

  };

}

#endif
