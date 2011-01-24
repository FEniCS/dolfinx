// Copyright (C) 2005-2011 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-02-13
// Last changed: 2011-01-24

#ifndef __INIT_H
#define __INIT_H

namespace dolfin
{

  /// Initialize DOLFIN (and PETSc) with command-line arguments. This
  /// should not be needed in most cases since the initialization is
  /// otherwise handled automatically.
  void init(int argc, char* argv[]);

}

#endif
