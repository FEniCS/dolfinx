// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __PARAMETER_LIST_H
#define __PARAMETER_LIST_H

#include <stdarg.h>

#include "Parameter.h"

namespace dolfin {
  
  class ParameterList{
  public:
	 
	 ParameterList();
	 ~ParameterList();
	 
	 /// Initialize and check all parameters
	 virtual void init() = 0;
	 
	 /// Add a parameter
	 void add(Parameter::Type type, const char *identifier, ...);
	 
	 /// Set the value of a parameter
	 void set(const char *identifier, ...);
	 void set_aptr(const char *identifier, va_list aptr);
	 
	 /// Get the value of a parameter
	 void get(const char *identifier, ...);
	 void get_aptr(const char *identifier, va_list aptr);
	 
	 /// Save all parameters to the default file
	 void save();
	 
	 /// Save all parameters to the given file
	 void save(const char *filename);
	 
	 /// Load all parameters from the default file
	 void load();
	 
	 /// Load all parameters from the given file
	 void load(const char *filename);
	 
	 /// Check if the parameter has been changed
	 bool changed(const char *identifier);
	 
  private:
	 
	 int  getIndex(const char *identifier);
	 void realloc();
	 
	 // A list of all the parameters
	 Parameter *parameters;
 
	 // Size of list
	 int alloc_size;

	 // Position for next item
	 int current;
	 
  };

}
  
#endif
