// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-05-06
// Last changed: 2005

#ifndef __PARAMETER_H
#define __PARAMETER_H

#include <string>
#include <stdarg.h>

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>
#include <dolfin/utils.h>

#define PARAMETER_IDENTIFIER_LENGTH 128

using std::string;

namespace dolfin
{

  // A small class for internal use in Settings
  class Parameter
  {
  public:
    
    enum Type { REAL, INT, BOOL, STRING, NONE };
    
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
    //operator const char*() const;

    // Friends
    friend class XMLFile;

    // Output
    friend LogStream& dolfin::operator<<(LogStream& stream, const Parameter& p);    

    // Type of data
    Type type;

    // A description of the parameter
    string identifier;
    
  private:
    
    // Values
    real       val_real;
    int        val_int;
    bool       val_bool;
    string     val_string;
    
    // True iff the default value was changed
    bool _changed;
    
    
  };
  
}

#endif
