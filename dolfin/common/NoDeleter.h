// Copyright (C) 2008 Garth N. Wells
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
// Modified by Martin Alnes, 2008.
//
// First added:  2008-09-11
// Last changed: 2011-01-28

#ifndef __NO_DELETER_H
#define __NO_DELETER_H

#include <memory>

namespace dolfin
{

  /// NoDeleter is a customised deleter intended for use with smart pointers.

  class NoDeleter
  {
  public:
    /// Do nothing
    void operator() (const void *) {}
  };

  /// Helper function to construct shared pointer with NoDeleter with cleaner syntax

  template <typename T>
  std::shared_ptr<T> reference_to_no_delete_pointer(T& r)
  {
    return std::shared_ptr<T>(&r, NoDeleter());
  }

}

#endif
