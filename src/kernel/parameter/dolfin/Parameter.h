// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-05-06
// Last changed: 2005-12-18

#ifndef __PARAMETER_H
#define __PARAMETER_H

#include <dolfin/dolfin_log.h>

namespace dolfin
{

  class ParameterValue;

  /// This class represents a parameter of some given type.
  /// Supported value types are real, int, bool, and string.

  class Parameter
  {
  public:

    /// Supported parameter types
    enum Type { type_real, type_int, type_bool, type_string, type_none };

    /// Create int-valued parameter
    Parameter(int value);

    /// Create real-valued parameter
    Parameter(real value);

    /// Create bool-valued parameter
    Parameter(bool value);

    /// Create string-valued parameter
    Parameter(std::string value);

    /// Create int-valued parameter from uint)
    Parameter(uint value);

    /// Copy constructor
    Parameter(const Parameter& parameter);

    /// Destructor
    ~Parameter();

    /// Assignment of int
    const Parameter& operator= (int value);

    /// Assignment of real
    const Parameter& operator= (real value);

    /// Assignment of bool
    const Parameter& operator= (bool value);

    /// Assignment of string
    const Parameter& operator= (std::string value);

    /// Assignment of uint
    const Parameter& operator= (uint value);

    /// Cast parameter to int
    operator int() const;

    /// Cast parameter to real
    operator real() const;

    /// Cast parameter to bool
    operator bool() const;

    /// Cast parameter to string
    operator std::string() const;

    /// Cast parameter to uint
    operator uint() const;

    /// Return type of parameter
    Type type() const;

    /// Friends
    friend class XMLFile;

  private:

    // Pointer to parameter value
    ParameterValue* value;

    // Type of parameter
    Type _type;
    
  };
  
}

#endif
