// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __PARAMETER_LIST_H
#define __PARAMETER_LIST_H

#include <stdarg.h>

#include <dolfin/ShortList.h>
#include "Parameter.h"

namespace dolfin {
  
  class ParameterList{
  public:
	 
	 /// Add a parameter
	 static void add(Parameter::Type type, const char *identifier, ...);
	 
	 /// Set the value of a parameter
	 static void set(const char *identifier, ...);
	 static void set_aptr(const char *identifier, va_list aptr);
	 
	 /// Get the value of a parameter
	 static void get(const char *identifier, ...);
	 static void get_aptr(const char *identifier, va_list aptr);
	 
	 /// Check if the parameter has been changed
	 static bool changed(const char *identifier);
	 
  private:
	 
	 static int getIndex(const char *identifier);
	 
	 static ShortList<Parameter> list;
	 
  };

}
  
#endif
