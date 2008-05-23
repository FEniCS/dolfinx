// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-23
// Last changed: 2008-05-23

#ifndef __NORMALIZATION_TYPE_H
#define __NORMALIZATION_TYPE_H

namespace dolfin
{

  /// Two different normalizations are available:
  /// 
  /// norm:    Normalizes a vector x according to x --> x / ||x||
  ///          where ||x|| is the l2 norm of x
  ///
  /// average: Normalizes a vector x according to x --> x - avg(x)
  ///          where avg(x) is the average of x. This is useful to
  ///          satisfy the compatibility condition for the Neumann
  ///          problem.

  enum NormalizationType {norm, average};

}

#endif
