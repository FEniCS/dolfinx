// Copyright (C) 2008 Garth N. Wells and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-08-24
// Last changed: 2008-08-27

#ifndef __ENUMS_LA_H
#define __ENUMS_LA_H

namespace dolfin
{

  /// List of norm types for vectors and matrices
  enum MatrixType {symmetric, nonsymmetric};

  /// Two different normalizations are available:
  ///
  /// normalize_average: Normalizes a vector x according to x --> x - avg(x)
  ///                    where avg(x) is the average of x. This is useful to
  ///                    satisfy the compatibility condition for the Neumann
  ///                    problem.
  ///
  /// normalize_l2norm:  Normalizes a vector x according to x --> x / ||x||
  ///                    where ||x|| is the l2 norm of x
  enum NormalizationType {normalize_average, normalize_l2norm};

}

#endif
