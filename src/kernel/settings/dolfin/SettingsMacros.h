// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// These are not really macros, but we try to mimic the structure
// of the log system.

#ifndef __SETTINGS_MACROS_H
#define __SETTINGS_MACROS_H

#include <stdarg.h>

namespace dolfin {

  // Add new parameter
  void dolfin_parameter(Parameter::Type type, const char *identifier, ...);

  // Set value of a parameter
  void dolfin_set(const char *identifier, ...);
  void dolfin_set_aptr(const char *identifier, va_list aptr);

  // Get value of a parameter
  Parameter dolfin_get(const char *identifier);

}

#endif
