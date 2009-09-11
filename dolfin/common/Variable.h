// Copyright (C) 2003-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-02-26
// Last changed: 2009-09-08

#ifndef __VARIABLE_H
#define __VARIABLE_H

#include <string>
#include <dolfin/parameter/Parameters.h>

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

    /// Destructor
    virtual ~Variable();

    /// Rename variable
    void rename(const std::string name, const std::string label);

    /// Return name
    const std::string& name()  const;

    /// Return label (description)
    const std::string& label() const;

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const;

    /// Deprecated, to be removed
    void disp() const;

    // Parameters
    Parameters parameters;

  private:

    // Name
    std::string _name;

    // Label
    std::string _label;

  };

}

#endif
