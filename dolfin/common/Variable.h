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
// Modified by Garth N. Wells, 2011
//
// First added:  2003-02-26
// Last changed: 2011-09-24

#ifndef __VARIABLE_H
#define __VARIABLE_H

#include <cstddef>
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

    /// Copy constructor
    Variable(const Variable& variable);

    /// Destructor
    virtual ~Variable();

    /// Assignment operator
    const Variable& operator=(const Variable& variable);

    /// Rename variable
    void rename(const std::string name, const std::string label);

    /// Return name
    std::string name()  const;

    /// Return label (description)
    std::string label() const;

    /// Get unique identifier.
    ///
    /// *Returns*
    ///     _std::size_t_
    ///         The unique integer identifier associated with the object.
    std::size_t id() const { return unique_id; }

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const;

    /// Parameters
    Parameters parameters;

  private:

    // Name
    std::string _name;

    // Label
    std::string _label;

    // Unique identifier
    const std::size_t unique_id;

  };

}

#endif
