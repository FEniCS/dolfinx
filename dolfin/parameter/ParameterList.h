// Copyright (C) 2003-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Benjamin Kehlet
// Modified by Ola Skavhaug
//
// First added:  2003-05-06
// Last changed: 2009-03-16

#ifndef __PARAMETER_LIST_H
#define __PARAMETER_LIST_H

#include <map>
#include "Parameter.h"

namespace dolfin
{

  class NewXMLParameterList;

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
    void add(std::string key, const Parameter& value);

    /// Set value of parameter
    void set(std::string key, const Parameter& value);

    /// Get value of parameter with given key
    Parameter get(std::string key) const;

    /// Check if parameter with given key has been defined
    bool defined(std::string key) const;

    /// Check if parameter has been changed
    bool changed(std::string key) const;

    /// Friends
    friend class XMLFile;
    friend class NewXMLParameterList;

    // Used by NewXMLFile for templated i/o
    typedef NewXMLParameterList XMLHandler;
 
  private:

    // Parameters stored as an STL map
    std::map<const std::string, Parameter> parameters;

    // Typedef of iterators for convenience
    typedef std::map<const std::string, Parameter>::iterator iterator;
    typedef std::map<const std::string, Parameter>::const_iterator const_iterator;

    // Typedef of pair for convenience
    typedef std::pair<const std::string, Parameter> pair;

  };

}

#endif
