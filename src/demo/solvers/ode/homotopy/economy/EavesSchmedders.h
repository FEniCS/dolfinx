// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EAVES_SCHMEDDERS_H
#define __EAVES_SCHMEDDERS_H

#include <dolfin.h>

using std::pow;

/// CES economy with two goods from Eaves' and Schmedders' "General
/// equilibrium models and homotopy methods" (1999).

/// This class implements the original rational form.

class EavesSchmedders : public Homotopy
{
public:

  EavesSchmedders() : Homotopy(2) {}

  void F(const complex z[], complex y[])
  {
    // First equation
    y[0] = z[0] + z[1] - 1.0;

    // Second equation
    const complex g0 = 4.0*pow(z[0], 0.8)*pow(z[1], 0.2) + z[1];
    const complex g1 = pow(z[0], 0.8)*pow(z[1], 0.2) + 4.0*z[1];
    y[1] = (10.0*z[0] + z[1]) / g0 + (4.0*z[0] + 48.0*z[1]) / g1 - 13.0;
  }

  void JF(const complex z[], const complex x[], complex y[])
  {
    // First equation
    y[0] = x[0] + x[1];
    
    // Second equation
    const complex g0 = 4.0*pow(z[0], 0.8)*pow(z[1], 0.2) + z[1];
    const complex g1 = pow(z[0], 0.8)*pow(z[1], 0.2) + 4.0*z[1];
    y[1] = ( ( (8.0*pow(z[0], 0.8)*pow(z[1], 0.2) + 10.0*z[1] - 
		3.2*pow(z[0], -0.2)*pow(z[1], 1.2)) / (g0*g0) +
	       (0.8*pow(z[0], 0.8)*pow(z[1], 0.2) + 16.0*z[1] -
		38.4*pow(z[0], -0.2)*pow(z[1], 1.2)) / (g1*g1) ) * x[1] +
	     ( (-8.0*pow(z[0], 1.8)*pow(z[1], -0.8) - 10.0*z[0] +
		3.2*pow(z[0], 0.8)*pow(z[1], 0.2)) / (g0*g0) +
	       (-0.8*pow(z[0], 1.8)*pow(z[1], -0.8) - 16.0*z[0] +
		38.4*pow(z[0], 0.8)*pow(z[1], 0.2)) / (g1*g1) ) * x[2] );
  }

  unsigned int degree(unsigned int i) const
  {
    if ( i == 0 )
      return 1;
    else
      return 2;
  }
  
};

/// This class implements the polynomial form.

class PolynomialEavesSchmedders : public Homotopy
{
public:

  PolynomialEavesSchmedders() : Homotopy(2) {}

  void F(const complex z[], complex y[])
  {
    // First equation
    y[0] = z[0] + z[1] - 1.0;
    
    // Second equation
    y[1] = ( 44.0 * z[0] * z[1] + 
	     26.0 * pow(z[0], 1.8) * pow(z[1], 0.2) -
	     28.0 * pow(z[0], 0.8) * pow(z[1], 1.2) -
	     52.0 * pow(z[0], 1.6) * pow(z[1], 0.4) );
  }

  void JF(const complex z[], const complex x[], complex y[])
  {
    // First equation
    y[0] = x[0] + x[1];

    // Second equation
    y[1] = ( (44.0 * z[1] +
	      46.8 * pow(z[0], 0.8)  * pow(z[1], 0.2) -
	      22.4 * pow(z[0], -0.2) * pow(z[1], 1.2) -
	      83.2 * pow(z[0], 0.6)  * pow(z[1], 0.4)) * x[0] +
	     (44.0 * z[0] +
	      5.2  * pow(z[0], 1.8)  * pow(z[1], -0.8) -
	      33.6 * pow(z[0], 0.8)  * pow(z[1], 1.2) -
	      20.8 * pow(z[0], 1.6)  * pow(z[1], -0.6)) * x[1] );
  }
  
  unsigned int degree(unsigned int i) const
  {
    if ( i == 0 )
      return 1;
    else
      return 2;
  }

};

#endif
