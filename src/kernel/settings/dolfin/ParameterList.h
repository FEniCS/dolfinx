// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-05-06
// Last changed: 2005

#ifndef __PARAMETER_LIST_H
#define __PARAMETER_LIST_H

#include <stdarg.h>

#include <dolfin/List.h>
#include <dolfin/Parameter.h>

namespace dolfin
{
  
  class ParameterList
  {
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

    /// Friends
    friend class XMLFile;
    
  private:
    
    Parameter* find(const char *identifier);
    
    List<Parameter> list;
    
  };
  
}

#endif
