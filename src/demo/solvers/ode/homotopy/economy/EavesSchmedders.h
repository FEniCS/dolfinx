// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EAVES_SCHMEDDERS_H
#define __EAVES_SCHMEDDERS_H

#include <dolfin.h>

/// CES economy with two goods from Eaves' and Schmedders' "General
/// equilibrium models and homotopy methods" (1999).

/// This class implements the original rational form.

class EavesSchmedders : public Homotopy
{
public:

  EavesSchmedders() : Homotopy(2) {}

  void F(const complex z[], complex y[])
  {
    dolfin_error("Not implemented.");
  }

  void JF(const complex z[], const complex x[], complex y[])
  {
    dolfin_error("Not implemented.");
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
	     26.0 * std::pow(z[0], 1.8) * std::pow(z[1], 0.2) -
	     28.0 * std::pow(z[0], 0.8) * std::pow(z[1], 1.2) -
	     52.0 * std::pow(z[0], 1.6) * std::pow(z[1], 0.4) );
  }

  void JF(const complex z[], const complex x[], complex y[])
  {
    // First equation
    y[0] = x[0] + x[1];

    // Second equation
    y[1] = ( (44.0 * z[1] +
	      46.8 * std::pow(z[0], 0.8)  * std::pow(z[1], 0.2) -
	      22.4 * std::pow(z[0], -0.2) * std::pow(z[1], 1.2) -
	      83.2 * std::pow(z[0], 0.6)  * std::pow(z[1], 0.4)) * x[0] +
	     (44.0 * z[0] +
	      5.2  * std::pow(z[0], 1.8)  * std::pow(z[1], -0.8) -
	      33.6 * std::pow(z[0], 0.8)  * std::pow(z[1], 1.2) -
	      20.8 * std::pow(z[0], 1.6)  * std::pow(z[1], -0.6)) * x[1] );
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
