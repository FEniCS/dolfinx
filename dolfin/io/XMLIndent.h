// Copyright (C) 2009 Ola Skavhaug
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
// First added:  2009-03-17
// Last changed: 2009-03-17

#ifndef __XMLINDENT_H
#define __XMLINDENT_H

#include <dolfin/common/types.h>

namespace dolfin
{
  /// This class is used for simplifying the indentation of xml files.
  // The operator ++ and -- increments and decrements the indentation level,
  // and operator() returns a string that represents the current indentation
  // level.

  class XMLIndent
  {
  public:

    /// Constructor
    XMLIndent(uint indentation_level, uint step_size=2);

    /// Destructor
    ~XMLIndent();

    void operator++();

    void operator--();

    std::string operator() () const;

    dolfin::uint level()
    { return indentation_level; }

  private:

    uint indentation_level;
    uint step_size;

  };

}
#endif
