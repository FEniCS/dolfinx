// Copyright (C) 2009 Benjamin Kehlet
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-02-01
// Last changed:

#ifndef __GMP_OBJECT_H
#define __GMP_OBJECT_H

#define BITS_PR_DIGIT 3.3299280948

#ifdef HAS_GMP
#include <gmpxx.h>
#endif

#include <dolfin/common/real.h>
#include <dolfin/parameter/parameters.h>

namespace dolfin
{ 
  class ODE;

  /// This class calls SubSystemsManger to initialise PETSc.
  ///
  /// All PETSc objects must be derived from this class.

  class GMPObject
  {
  public:

    GMPObject() { 
#ifdef HAS_GMP
      //compute the number of bits needed
      uint decimal_prec = dolfin_get("floating-point precision");
      mpf_set_default_prec( (uint) (decimal_prec*BITS_PR_DIGIT));
      
      real eps = real_epsilon();
      dolfin_set("ODE discrete tolerance", to_double(eps*10));
      char msg[100];
      gmp_sprintf(msg, "Epsilon=%Fe", eps.get_mpf_t());
      message(msg);
      message("GMP: Using %d bits pr number", mpf_get_default_prec());
#else 
      if (dolfin_changed("floating-point precision")) {
	warning("Can't change floating-point precision when using type double");
      }
#endif
    }
  };

}

#endif
