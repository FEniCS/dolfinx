// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TIMEINFO_H
#define __TIMEINFO_H

#include <dolfin/constants.h>

namespace dolfin {

  const char* date();

  void tic();
  real toc();

}

#endif
