// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __DOLFIN_CONSTANTS_H
#define __DOLFIN_CONSTANTS_H

#include <float.h>

#define DOLFIN_VERSION       PACKAGE_VERSION
#define DOLFIN_LINELENGTH    1024
#define DOLFIN_WORDLENGTH    128
#define DOLFIN_PARAMSIZE     32
#define DOLFIN_TERM_WIDTH    80
#define DOLFIN_EPS           3.0e-16
#define DOLFIN_SQRT_EPS      1.0e-8
#define DOLFIN_PI            3.141592653589793238462
#define DOLFIN_PROGRESS_BARS 4
#define DOLFIN_PROGRESS_WAIT 2
#define DOLFIN_BLOCK_SIZE    1024
#define DOLFIN_MEGABYTE      1048576

// FIXME: Maybe we should put this somewhere else?
namespace dolfin
{
  
  typedef double real;
  
}

#endif
