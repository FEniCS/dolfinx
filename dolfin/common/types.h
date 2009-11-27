// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-04-22
// Last changed: 2009-11-27
//
// This file provides DOLFIN typedefs for basic types.

#ifndef __DOLFIN_TYPES_H
#define __DOLFIN_TYPES_H

#include <complex>
#include <set>

namespace dolfin
{

  // Unsigned integers
  typedef unsigned int uint;

  // Complex numbers
  typedef std::complex<double> complex;

  // (Ordered) set of uints
  typedef std::set<uint> uint_set; 

}

#endif
