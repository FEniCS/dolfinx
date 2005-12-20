// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-12-19
// Last changed: 2005-12-19

#ifndef __PARAMETER_SYSTEM_H
#define __PARAMETER_SYSTEM_H

#include <dolfin/ParameterList.h>

namespace dolfin
{

  /// This class holds a global database of parameters for DOLFIN,
  /// implemented as a set of (key, value) pairs. Supported value
  /// types are real, int, bool, and string.

  class ParameterSystem : public ParameterList
  {
  public:

    /// Singleton instance of global parameter database
    static ParameterSystem parameters;
    
    /// Friends
    friend void add(std::string key, Parameter value);
    friend void set(std::string key, Parameter value);
    friend Parameter get(std::string key);

  private:

    // Constructor
    ParameterSystem();
    
  };

  /// The the basic functionality of the parameter system is exported
  /// to the dolfin namespace, including the three global functions
  /// add(), set() and get().
  
  /// Add parameter
  void add(std::string key, Parameter value);
  
  /// Set value of parameter
  void set(std::string key, Parameter value);
  
  /// Get value of parameter with given key
  Parameter get(std::string key);
  
}

#endif
