// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Contains small nonspecific utility functions useful for various
// tasks, such as string manipulation, simple type definitions, ...

#ifndef __UTILS_H
#define __UTILS_H

#include <stdio.h>
#include <dolfin/constants.h>

namespace dolfin {
  
  bool suffix         (const char *string, const char *suffix);
  void remove_newline (char *string);
  int  length         (const char *string);
  
  void delay(real seconds);
  
}
  
#endif
