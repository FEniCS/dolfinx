// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-08-19
// Last changed: 2008-08-19

#ifndef __NORMALIZATION_TYPE_H
#define __NORMALIZATION_TYPE_H

namespace dolfin
{

  /// Two different normalizations are available:
  /// 
  /// normalize_l2norm:  Normalizes a vector x according to x --> x / ||x||
  ///                    where ||x|| is the l2 norm of x
  ///
  /// normalize_average: Normalizes a vector x according to x --> x - avg(x)
  ///                    where avg(x) is the average of x. This is useful to
  ///                    satisfy the compatibility condition for the Neumann
  ///                    problem.

  enum NormalizationType {normalize_average, normalize_l2norm};

}

#endif
