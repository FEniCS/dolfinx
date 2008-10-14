// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Benjamin Kehlet, 2008.
//
// First added:  2008-10-02
// Last changed: 2008-10-14
//
// This file provides utilities for working with variable-precision
// floating-point numbers. It defines a datatype "real" which defaults
// to double, but may be changed to higher-precision representations,
// for example using GMP.

#ifndef __REAL_H
#define __REAL_H

#ifdef HAS_GMP
#include <gmpxx.h>
#endif

#include "types.h"
#include <dolfin/log/log.h>
#include <cmath>
#include <iostream>

namespace dolfin
{

  // Real type
#ifdef HAS_GMP
  typedef mpf_class real;
#else
  typedef double real;
#endif

  // Convert to double (if necessary)
  inline double to_double(const real x)
  {
#ifdef HAS_GMP 
    return x.get_d();
#else
    return x;
#endif
  }
  
  // Absolute value
  inline real abs (real x) 
  { return x >= 0 ? x : -1*y; }
  
  // Maximum
  inline real max (real x, real y)
  { return x > y ? x : y; }
  
  // Minimum
  inline real min (real x, real y)
  { return x < y ? x : y; }
    
  // Power function
  inline real pow(const real x, uint y) 
  { 
#ifdef HAS_GMP
    real res;
    mpf_pow_ui(res.get_mpf_t(), x.get_mpf_t(), y);
    return res;
#else
    return std::pow(to_double(x), to_double(y));
#endif
  }

  // Set array to given array (copy values)
  inline void real_set(uint n, real* x, const real* y)
  { for (uint i = 0; i < n; i++) x[i] = y[i]; }

  // Set array to given number
  inline void real_set(uint n, real* x, real value)
  { for (uint i = 0; i < n; i++) x[i] = value; }
  
  // Set array to zero
  inline void real_zero(uint n, real* x)
  { for (uint i = 0; i < n; i++) x[i] = 0.0; }
  
  // Add array, x += y
  inline void real_add(uint n, real* x, const real* y)
  { for (uint i = 0; i < n; i++) x[i] += y[i]; }
  
  // Subtract array, x -= y
  inline void real_sub(uint n, real* x, const real* y)
  { for (uint i = 0; i < n; i++) x[i] -= y[i]; }
  
  // Add multiple of array, x += a*y (AXPY operation)
  inline void real_axpy(uint n, real* x, real a, const real* y)
  { for (uint i = 0; i < n; i++) x[i] += a*y[i]; }
  
  // Multiply array with given number
  inline void real_mult(uint n, real* x, real a)
  { for (uint i = 0; i < n; i++) x[i] *= a; }
  
  // Compute inner products of array
  inline real real_inner(uint n, const real* x, const real* y)
  { real sum = 0.0; for (uint i = 0; i < n; i++) sum += x[i]*y[i]; return sum; }
  
  // Compute maximum absolute value of array
  inline real real_max_abs(uint n, const real* x)
  { real _max = 0.0; for (uint i = 0; i < n; i++) _max = max(abs(x[i]), _max); return _max; }
  
}

#endif
