// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-05-06
// Last changed: 2005-12-18

#ifndef __PARAMETER_LIST_H
#define __PARAMETER_LIST_H

#include <map>
#include <dolfin/Parameter.h>

namespace dolfin
{
  
  /// This class represents a database of parameters, where each
  /// parameter is uniquely identified by a string.
  
  class ParameterList
  {
  public:

    /// Constructor
    ParameterList();

    /// Destructor
    ~ParameterList();
	 
    /// Add parameter
    void add(std::string key, Parameter value);
    
    /// Set value of parameter
    void set(std::string key, Parameter value);

    /// Get value of parameter with given key
    Parameter get(std::string key);

    /// Check if parameter with given key has been defined
    bool defined(std::string key);

    /// Friends
    friend class XMLFile;
    
  private:

    // Parameters stored as an STL map
    std::map<std::string, Parameter> parameters;

    // Typedef of iterator for convenience
    typedef std::map<std::string, Parameter>::iterator iterator;
    
    // Typedef of pair for convenience
    typedef std::pair<std::string, Parameter> pair;
    
  };
  
}

#endif
