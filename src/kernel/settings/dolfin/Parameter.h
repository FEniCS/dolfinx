// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __PARAMETER_H
#define __PARAMETER_H

#include <bits/stl_alloc.h>
#include <string>
#include <stdarg.h>

#include <dolfin/constants.h>
#include <dolfin/function.h>
#include <dolfin/vfunction.h>
#include <dolfin/bcfunction.h>
#include <dolfin/utils.h>

#define PARAMETER_IDENTIFIER_LENGTH 128

using std::string;

namespace dolfin {

  // A small class for internal use in Settings
  class Parameter{
  public:
    
    enum Type { REAL, INT, BOOL, STRING, FUNCTION, VFUNCTION, BCFUNCTION, NONE };
    
    Parameter();
    Parameter(Type type, const char *identifier, va_list aptr);
    
    ~Parameter();
    
    void clear();
    
    void set(Type type, const char *identifier, va_list aptr);
    void set(const char *identifier, va_list aptr);
    void get(va_list aptr);
    
    bool matches(const char* identifier);
    bool matches(string identifier);
    bool changed();
    
    void operator= (const Parameter &p);
    
    // Needed for ShortList
    void operator= (int zero);
    bool operator! () const;

    // Type cast, enable assignment to type from Parameter
    operator real() const;
    operator int() const;
    operator unsigned int() const;
    operator bool() const;
    operator string() const;
    operator const char*() const;
    operator function() const;
    operator vfunction() const;
    operator bcfunction() const;

    // Output
    friend LogStream& dolfin::operator<<(LogStream& stream, const Parameter& p);
    
  private:
    
    // A description of the parameter
    string identifier;
    
    // Values
    real       val_real;
    int        val_int;
    bool       val_bool;
    string     val_string;
    function   val_function;
    vfunction  val_vfunction;
    bcfunction val_bcfunction;
    
    // True iff the default value was changed
    bool _changed;
    
    // Type of data
    Type type;
    
  };
  
}

#endif
