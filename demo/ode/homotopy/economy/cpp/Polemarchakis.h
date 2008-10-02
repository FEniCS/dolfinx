// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-04-01
// Last changed: 2005

#ifndef __POLEMARHCAKIS_H
#define __POLEMARHCAKIS_H

#include <dolfin.h>

/// Economy with two goods from Polemarchakis's "On the transfer paradox".
/// This class implements the original rational form.

class Polemarchakis : public Homotopy
{
public:

  Polemarchakis() : Homotopy(2), lambda(0), gamma(0)
  {
    lambda = new double[3];
    gamma = new double[3];

    lambda[0] = 7.0/5.0; lambda[1] = 7.0/10.0; lambda[2] = 13.0/5.0;
    gamma[0]= -22.0/5.0; gamma[1] = 13.0/5.0;  gamma[2]  = 93.0/85.0;
  }

  ~Polemarchakis()
  {
    if ( lambda ) delete [] lambda;
    if ( gamma ) delete [] gamma;
  }

  void F(const complex z[], complex y[])
  {
    // First equation
    complex sum = 0.0;
    for (unsigned int i = 0; i < 3; i++)
      sum += gamma[i] / (z[0] + lambda[i]);
    y[0] = sum;

    // Second equation
    y[1] = z[1] - 1.0;
  }

  void JF(const complex z[], const complex x[], complex y[])
  {
    // First equation
    complex sum = 0.0;
    for (unsigned int i = 0; i < 3; i++)
    {
      const complex tmp0 = z[0] + lambda[i];
      sum -= gamma[i] / (tmp0*tmp0);
    }
    y[0] = sum * x[0];
    
    // Second equation
    y[1] = x[1];
  }

  unsigned int degree(unsigned int i) const
  {
    if ( i == 0 )
      return 2;
    else
      return 1;
  }
  
private:

  double* lambda;
  double* gamma;

};

/// Economy with two goods from Polemarchakis's "On the transfer paradox".
/// This class implements the polynomial form.

class PolynomialPolemarchakis : public Homotopy
{
public:

  PolynomialPolemarchakis() : Homotopy(2), lambda(0), gamma(0)
  {
    lambda = new double[3];
    gamma = new double[3];

    lambda[0] = 7.0/5.0; lambda[1] = 7.0/10.0; lambda[2] = 13.0/5.0;
    gamma[0]= -22.0/5.0; gamma[1] = 13.0/5.0;  gamma[2]  = 93.0/85.0;
  }

  ~PolynomialPolemarchakis()
  {
    if ( lambda ) delete [] lambda;
    if ( gamma ) delete [] gamma;
  }

  void F(const complex z[], complex y[])
  {
    // First equation
    y[0] = ( gamma[0]*(z[0] + lambda[1])*(z[0] + lambda[2]) +
	     gamma[1]*(z[0] + lambda[0])*(z[0] + lambda[2]) +
	     gamma[2]*(z[0] + lambda[0])*(z[0] + lambda[1]) );
    
    // Second equation
    y[1] = z[1] - 1.0;
  }

  void JF(const complex z[], const complex x[], complex y[])
  {
    // First equation
    y[0] = ( gamma[0]*(2.0*z[0] + lambda[1] + lambda[2]) + 
	     gamma[1]*(2.0*z[0] + lambda[0] + lambda[2]) + 
	     gamma[2]*(2.0*z[0] + lambda[0] + lambda[1]) ) * x[0];
    
    // Second equation
    y[1] = x[1];
  }
  
  unsigned int degree(unsigned int i) const
  {
    if ( i == 0 )
      return 2;
    else
      return 1;
  }
  
private:

  double* lambda;
  double* gamma;

};

#endif
