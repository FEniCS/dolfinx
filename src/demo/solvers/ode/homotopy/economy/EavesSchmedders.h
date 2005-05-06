// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EAVES_SCHMEDDERS_H
#define __EAVES_SCHMEDDERS_H

#include <dolfin.h>

using std::pow;

/// CES economy with two goods from Eaves' and Schmedders' "General
/// equilibrium models and homotopy methods" (1999).

/// This class implements the original rational form with rational exponents.

class RationalRationalES : public Homotopy
{
public:

  RationalRationalES() : Homotopy(2) {}

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
		38.4*pow(z[0], -0.2)*pow(z[1], 1.2)) / (g1*g1) ) * x[0] +
	     ( (-8.0*pow(z[0], 1.8)*pow(z[1], -0.8) - 10.0*z[0] +
		3.2*pow(z[0], 0.8)*pow(z[1], 0.2)) / (g0*g0) +
	       (-0.8*pow(z[0], 1.8)*pow(z[1], -0.8) - 16.0*z[0] +
		38.4*pow(z[0], 0.8)*pow(z[1], 0.2)) / (g1*g1) ) * x[1] );
  }

  unsigned int degree(unsigned int i) const
  {
    if ( i == 0 )
      return 1;
    else
      return 2;
  }
  
};

/// This class implements the polynomial form with rational exponents.

class PolynomialRationalES : public Homotopy
{
public:

  PolynomialRationalES() : Homotopy(2) {}

  void F(const complex z[], complex y[])
  {
    // First equation
    y[0] = z[0] + z[1] - 1.0;
    
    // Second equation
    y[1] = ( 26.0 * pow(z[0], 1.8) * pow(z[1], 0.2) -
	     52.0 * pow(z[0], 1.6) * pow(z[1], 0.4) +
	     44.0 * z[0] * z[1] -
	     28.0 * pow(z[0], 0.8) * pow(z[1], 1.2) );

    // Regularization
    y[1] += (z[0] + z[1]) * (z[0] + z[1]) - 1.0;
    //y[1] += (z[0] + z[1]) * (z[0] + z[1]) * (z[0] + z[1]) - 1.0;
  }

  void JF(const complex z[], const complex x[], complex y[])
  {
    // First equation
    y[0] = x[0] + x[1];

    // Second equation
    y[1] = ( (46.8 * pow(z[0], 0.8)  * pow(z[1], 0.2) -
	      83.2 * pow(z[0], 0.6)  * pow(z[1], 0.4) +
	      44.0 * z[1] -
	      22.4 * pow(z[0], -0.2) * pow(z[1], 1.2)) * x[0] +
	     (5.2  * pow(z[0], 1.8)  * pow(z[1], -0.8) -
	      20.8 * pow(z[0], 1.6)  * pow(z[1], -0.6) +
	      44.0 * z[0] -
      	      33.6 * pow(z[0], 0.8)  * pow(z[1], 0.2)) * x[1] );

    // Regularization
    y[1] += 2.0 * (z[0] + z[1]) * (x[0] + x[1]);
    //y[1] += 3.0 * (z[0] + z[1]) * (z[0] + z[1]) * (x[0] + x[1]);
  }
  
  unsigned int degree(unsigned int i) const
  {
    if ( i == 0 )
      return 5;
    else
      return 9;
  }

};

/// This class implements the rational form with integer coefficients
/// for (z1^0.2, z2^0.2) --> (z1, z2)

class RationalIntegerES : public Homotopy
{
public:

  RationalIntegerES() : Homotopy(2) {}

  void F(const complex z[], complex y[])
  {
    // First equation
    y[0] = pow(z[0], 5) + pow(z[1], 5) - 1.0;
    
    // Second equation
    const complex g0 = 4.0*pow(z[0], 4)*z[1] + pow(z[1], 5);
    const complex g1 = pow(z[0], 4)*z[1] + 4.0*pow(z[1], 5);
    y[1] = ( (10.0*pow(z[0], 5) + pow(z[1], 5)) / g0 +
	     (4.0*pow(z[0], 5) + 48.0*pow(z[1], 5)) / g1 - 13.0 );
  }

  void JF(const complex z[], const complex x[], complex y[])
  {
    // First equation
    y[0] = 5.0 * pow(z[0], 4) * x[0] + 5.0 * pow(z[1], 4) * x[1];

    // Second equation
    const complex g0 = 4.0*pow(z[0], 4)*z[1] + pow(z[1], 5);
    const complex g1 = pow(z[0], 4)*z[1] + 4.0*pow(z[1], 5);
    y[1] = ( ( (50.0*pow(z[0], 4)*(4.0*pow(z[0], 4)*z[1] + pow(z[1], 5)) - 
		(10.0*pow(z[0], 5) + pow(z[1], 5))*16.0*pow(z[0], 3)*z[1]) / (g0*g0) +
	       (20.0*pow(z[0], 4)*(pow(z[0], 4)*z[1] + 4.0*pow(z[1], 5)) -
		(4.0*pow(z[0], 5) + 48.0*pow(z[1], 5))*4.0*pow(z[2], 3)*z[1]) / (g1*g1)) * x[0] +
	     ( (5.0*pow(z[1], 4)*(4.0*pow(z[0], 4)*z[1] + pow(z[1], 5)) -
		(10.0*pow(z[0], 5) + pow(z[1], 5))*(4.0*pow(z[0], 4) + 5.0*pow(z[1], 4))) / (g0*g0) +
	       (240.0*pow(z[1], 4)*(pow(z[0], 4)*z[1] + 4.0*pow(z[1], 5)) -
		(4.0*pow(z[0], 5) + 48.0*pow(z[1], 5))*(pow(z[0], 4) + 20.0*pow(z[1], 4))) / (g1*g1)) * x[1] );
  }
  
  unsigned int degree(unsigned int i) const
  {
    if ( i == 0 )
      return 5;
    else
      return 9;
  }

};

/// This class implements the polynomial form with integer coefficients
/// for (z1^0.2, z2^0.2) --> (z1, z2)

class PolynomialIntegerES : public Homotopy
{
public:

  PolynomialIntegerES() : Homotopy(2) {}

  void F(const complex z[], complex y[])
  {
    // First equation
    y[0] = pow(z[0], 5) + pow(z[1], 5) - 1.0;
    
    // Second equation
    y[1] = ( 26.0 * pow(z[0], 9) * z[1] -
	     52.0 * pow(z[0], 8) * pow(z[1], 2) +
	     44.0 * pow(z[0], 5) * pow(z[1], 5) -
	     28.0 * pow(z[0], 4) * pow(z[1], 6) );
  }

  void JF(const complex z[], const complex x[], complex y[])
  {
    // First equation
    y[0] = 5.0 * pow(z[0], 4) * x[0] + 5.0 * pow(z[1], 4) * x[1];

    // Second equation
    y[1] = ( (234.0 * pow(z[0], 8) * z[1] -
	      416.0 * pow(z[0], 7) * pow(z[1], 2) +
	      220.0 * pow(z[0], 4) * pow(z[1], 5) -
	      112.0 * pow(z[0], 3) * pow(z[1], 6)) * x[0] +
	     ( 26.0 * pow(z[0], 9) -
	      104.0 * pow(z[0], 8) * z[1] +
	      220.0 * pow(z[0], 5) * pow(z[1], 4) -
      	      168.0 * pow(z[0], 4) * pow(z[1], 5)) * x[1] );
  }
  
  unsigned int degree(unsigned int i) const
  {
    if ( i == 0 )
      return 5;
    else
      return 9;
  }

};

#endif
