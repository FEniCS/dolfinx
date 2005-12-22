// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-02-06
// Last changed: 2005-12-21
//
// Contains small nonspecific utility functions useful for various
// tasks, such as string manipulation, simple type definitions, ...

#ifndef __UTILS_H
#define __UTILS_H

#include <string>
#include <dolfin/constants.h>

namespace dolfin
{
  
  bool suffix         (const char *string, const char *suffix);
  void remove_newline (char *string);
  int  length         (const char *string);

  std::string date();
  void delay(real seconds);
  
}
  
#endif
