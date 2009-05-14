// Copyright (C) 2009 Benjamin Kehlet
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2009.
//
// First added:  2009-02-01
// Last changed: 2009-02-09

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

  /// This class calls SubSystemsManger to initialise GMP.

  class GMPObject : public Parametrized
  {
  public:

    GMPObject()
    {
#ifdef HAS_GMP
      // Compute the number of bits needed
      const uint decimal_prec = dolfin_get("floating-point precision");
      mpf_set_default_prec(static_cast<uint>(decimal_prec*BITS_PR_DIGIT));

      // Compute epsilon
      real_init();

      // Set the default discrete tolerance
      set("ODE discrete tolerance", to_double(real_sqrt(real_epsilon())));

      // Display number of digits
      char msg[100];
      gmp_sprintf(msg, "%Fe", real_epsilon().get_mpf_t());
      info("Using %d bits per digit, eps = %s", mpf_get_default_prec(), msg);

#else
      if (dolfin_changed("floating-point precision"))
	warning("Can't change floating-point precision when using type double");
#endif
    }
  };

}

#endif
