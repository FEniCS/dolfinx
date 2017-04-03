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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2010-10-11
// Last changed: 2012-09-03

#ifndef __MARKING_H
#define __MARKING_H

#include <string>

namespace dolfin
{

  class Vector;
  template <typename T> class MeshFunction;

  /// Mark cells based on indicators and given marking strategy
  ///
  /// @param    markers (MeshFunction<bool>)
  ///         the cell markers (to be computed)
  ///
  /// @param    indicators (MeshFunction<double>)
  ///         error indicators (one per cell)
  ///
  /// @param    strategy (std::string)
  ///         the marking strategy
  ///
  /// @param    fraction (double)
  ///         the marking fraction
  void mark(MeshFunction<bool>& markers,
            const dolfin::MeshFunction<double>& indicators,
            const std::string strategy,
            const double fraction);

  /// Mark cells using Dorfler marking
  ///
  /// @param    markers (MeshFunction<bool>)
  ///         the cell markers (to be computed)
  ///
  /// @param    indicators (MeshFunction<double>)
  ///         error indicators (one per cell)
  ///
  /// @param    fraction (double)
  ///         the marking fraction
  void dorfler_mark(MeshFunction<bool>& markers,
                    const dolfin::MeshFunction<double>& indicators,
                    const double fraction);

}

#endif
