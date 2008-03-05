// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-02-13
// Last changed: 2005

#ifndef __INIT_H
#define __INIT_H

namespace dolfin
{
  
  /// Initialize DOLFIN (and PETSc) with command-line arguments. This
  /// should not be needed in most cases since the initialization is
  /// otherwise handled automatically.
  void dolfin_init(int argc, char* argv[]);

}

#endif
