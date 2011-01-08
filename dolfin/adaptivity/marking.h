// Copyright (C) 2010 Marie E. Rognes
// Licensed under the GNU LGPL Version 3.0 or any later version
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
