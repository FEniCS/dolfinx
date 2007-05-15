// Copyright (C) 2002-2007 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2002-12-16
// Last changed: 2007-05-15

#ifndef __DOLFIN_CONSTANTS_H
#define __DOLFIN_CONSTANTS_H

#include <complex>

// DOLFIN constants
#define DOLFIN_VERSION       PACKAGE_VERSION
#define DOLFIN_EPS           3.0e-16
#define DOLFIN_SQRT_EPS      1.0e-8
#define DOLFIN_PI            3.141592653589793238462
#define DOLFIN_LINELENGTH    256
#define DOLFIN_TERM_WIDTH    80

// DOLFIN typedefs
namespace dolfin
{
  
  typedef double real;
  typedef unsigned int uint;
  typedef std::complex<double> complex;

}

#endif
