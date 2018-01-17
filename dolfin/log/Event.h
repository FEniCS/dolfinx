// Copyright (C) 2004-2007 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

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
  void operator()();

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


