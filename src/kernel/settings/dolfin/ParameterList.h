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

    ParameterList();
	 
    /// Add a parameter
    void add(Parameter::Type type, const char *identifier, ...);
    void add_aptr(Parameter::Type type, const char *identifier, va_list aptr);
    
    /// Set the value of a parameter
    void set(const char *identifier, ...);
    void set_aptr(const char *identifier, va_list aptr);
    
    /// Get the value of a parameter
    Parameter get(const char *identifier);
    
    /// Check if the parameter has been changed
    bool changed(const char *identifier);
    
    /// Check if the list is empty
    bool empty();
    
  private:
    
    int getIndex(const char *identifier);
    
    ShortList<Parameter> list;
    bool _empty;
    
  };
  
}

#endif
