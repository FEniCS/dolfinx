// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

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
