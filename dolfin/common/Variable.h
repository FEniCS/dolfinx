// Copyright (C) 2003-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-02-26
// Last changed: 2009-05-09

#ifndef __VARIABLE_H
#define __VARIABLE_H

#include <string>

namespace dolfin
{

  /// Common base class for DOLFIN variables.
  
  class Variable
  {
  public:
    
    /// Create unnamed variable
    Variable();

    /// Create variable with given name and label
    Variable(const std::string name, const std::string label);

    /// Rename variable
    void rename(const std::string name, const std::string label);
    
    /// Return name
    const std::string& name()  const;

    /// Return label (description)
    const std::string& label() const;

    /// Return informal string representation (pretty-print)
    virtual std::string str() const;
    
  private:

    // Name
    std::string _name;

    // Label
    std::string _label;

  };

}

#endif
