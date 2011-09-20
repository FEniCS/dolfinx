// Copyright (C) 2003-2009 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2003-02-26
// Last changed: 2011-09-20

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
