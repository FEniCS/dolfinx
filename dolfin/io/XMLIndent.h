// Copyright (C) 2009 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
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

    std::string operator() ();

    inline dolfin::uint level() { return indentation_level; }

  private:
    uint indentation_level;
    uint step_size;

  };

}
#endif
