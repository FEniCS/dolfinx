// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FUNCTION_H
#define __FUNCTION_H

#include <dolfin/Grid.h>

namespace dolfin {

  class Function {
  public:

	 Function(Grid &grid_);
	 ~Function();

  private:

	 Grid &grid;
	 
  };

}

#endif
