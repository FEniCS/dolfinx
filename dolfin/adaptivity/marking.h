// Copyright (C) 2010 Marie E. Rognes
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2010-10-11
// Last changed: 2010-11-30

#ifndef __MARKING_H
#define __MARKING_H

#include <string>

namespace dolfin
{

  class Vector;
  template <class T> class MeshFunction;

  /// Mark cells based on indicators and given marking strategy
  ///
  /// *Arguments*
  ///     markers (_MeshFunction<bool>_)
  ///         the cell markers (to be computed)
  ///
  ///     indicators (_Vector_)
  ///         error indicators (one per cell)
  ///
  ///     strategy (std::string)
  ///         the marking strategy
  ///
  ///     fraction (double)
  ///         the marking fraction
  void mark(MeshFunction<bool>& markers, const Vector& indicators,
            const std::string strategy, const double fraction);

  /// Mark cells using Dorfler marking
  ///
  /// *Arguments*
  ///     markers (_MeshFunction<bool>_)
  ///         the cell markers (to be computed)
  ///
  ///     indicators (_Vector_)
  ///         error indicators (one per cell)
  ///
  ///     fraction (double)
  ///         the marking fraction
  void dorfler_mark(MeshFunction<bool>& markers, const Vector& indicators,
                    const double fraction);

}

#endif
