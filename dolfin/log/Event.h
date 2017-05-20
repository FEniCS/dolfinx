// Copyright (C) 2004-2007 Anders Logg
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
// First added:  2004-01-03
// Last changed: 2007-05-14

#ifndef __EVENT_H
#define __EVENT_H

#include <string>

namespace dolfin
{

  /// A event is a string message which is displayed
  /// only a limited number of times.
  ///
  /// @code{.cpp}
  ///
  ///         Event event("System is stiff, damping is needed.");
  ///         while ()
  ///         {
  ///           ...
  ///           if ( ... )
  ///           {
  ///             event();
  ///             ...
  ///           }
  ///         }
  /// @endcode

  class Event
  {
  public:

    /// Constructor
    Event(const std::string msg, unsigned int maxcount = 1);

    /// Destructor
    ~Event();

    /// Display message
    void operator() ();

    /// Display count
    unsigned int count() const;

    /// Maximum display count
    unsigned int maxcount() const;

  private:

    std::string _msg;
    unsigned int _maxcount;
    unsigned int _count;

  };

}

#endif
