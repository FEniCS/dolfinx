// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __LEONTIEF_H
#define __LEONTIEF_H

#include <dolfin.h>
#include "Economy.h"

using namespace dolfin;

// Leontief economy (rational form)

class Leontief : public Economy
{
public:

  Leontief(unsigned int m, unsigned int n) : Economy(m, n)
  {
    // Special choice of data
    a[0][0] = 2.0; a[0][1] = 1.0;
    a[1][0] = 1.0; a[1][1] = 2.0;
    
    w[0][0] = 1.0; w[0][1] = 2.0;
    w[1][0] = 2.0; w[1][1] = 1.0;

    init(&tmp0);
    init(&tmp1);
  }

  void F(const complex z[], complex y[])
  {
    // First equation: normalization
    y[0] = sum(z) - 1.0;

    // Precompute scalar products
    for (unsigned int i = 0; i < m; i++)
      tmp0[i] = dot(w[i], z) / dot(a[i], z);
    
    // Evaluate right-hand side
    for (unsigned int j = 1; j < n; j++)
    {
      complex sum = 0.0;
      for (unsigned int i = 0; i < m; i++)
	sum += a[i][j] * tmp0[i] - w[i][j];
      y[j] = sum;
    }
  }

  void JF(const complex z[], const complex x[], complex y[])
  {
    // First equation: normalization
    y[0] = sum(x);

    // Precompute scalar products
    for (unsigned int i = 0; i < m; i++)
    {
      const complex az = dot(a[i], z);
      const complex wz = dot(w[i], z);
      const complex ax = dot(a[i], x);
      const complex wx = dot(w[i], x);
      tmp0[i] = (az*wx - wz*ax) / (az*az);
    }
    
    // Evaluate right-hand side
    for (unsigned int j = 1; j < n; j++)
    {
      complex sum = 0.0;
      for (unsigned int i = 0; i < m; i++)
	sum += a[i][j] * tmp0[i];
      y[j] = sum;
    }
  }

  unsigned int degree(unsigned int i) const
  {
    if ( i == 0 )
      return 1;
    else
      return m;
  }
  
};

// Leontief economy (polynomial form)

class PolynomialLeontief : public Economy
{
public:

  PolynomialLeontief(unsigned int m, unsigned int n) : Economy(m, n)
  {
    // Special choice of data
    a[0][0] = 2.0; a[0][1] = 1.0;
    a[1][0] = 1.0; a[1][1] = 2.0;

    w[0][0] = 1.0; w[0][1] = 2.0;
    w[1][0] = 2.0; w[1][1] = 1.0;
  }

  void F(const complex z[], complex y[])
  {
    // First equation: normalization
    y[0] = sum(z) - 1.0;

    // Precompute scalar products a*z
    for (unsigned int i = 0; i < m; i++)
      tmp0[i] = dot(a[i], z);
    
    // Precompute special products
    for (unsigned int i = 0; i < m; i++)
    {
      complex tmp = dot(w[i], z);
      for (unsigned int k = 0; k < m; k++)
	if ( k != i )
	  tmp *= tmp0[k];
      tmp1[i] = tmp;
    }
    
    // Precompute special product
    complex tmp = 1.0;
    for (unsigned int i = 0; i < m; i++)
      tmp *= tmp0[i];
    
    // Evaluate right-hand side
    for (unsigned int j = 1; j < n; j++)
    {
      complex sum = 0.0;
      for (unsigned int i = 0; i < m; i++)
	  sum += a[i][j] * tmp1[i] - w[i][j] * tmp;
      y[j] = sum;
    }
  }

  void JF(const complex z[], const complex x[], complex y[])
  {
    // First equation: normalization
    y[0] = sum(x);

    // Precompute scalar products a*z
    for (unsigned int i = 0; i < m; i++)
      tmp0[i] = dot(a[i], z);
    
    // Evaluate first term
    for (unsigned int i = 0; i < m; i++)
    {
      complex tmp = dot(w[i], x);
      for (unsigned int k = 0; k < m; k++)
	  if ( k != i )
	    tmp *= tmp0[k];
      tmp1[i] = tmp;
    }
    for (unsigned int j = 1; j < n; j++)
    {
      complex sum = 0.0;
      for (unsigned int i = 0; i < m; i++)
	sum += a[i][j] * tmp1[i];
      y[j] = sum;
    }
    
    // Evaluate second term
    for (unsigned int i = 0; i < m; i++)
    {
      complex sum = 0.0;
      for (unsigned int k = 0; k < m; k++)
      {
	if ( k != i )
	{
	  complex product = 1.0;
	  for (unsigned int r = 0; r < m; r++)
	    if ( r != i && r != k )
	      product *= tmp0[r];
	  sum += dot(a[k], x);
	}
      }
      tmp1[i] = dot(w[i], z) * sum;
    }
    for (unsigned int j = 1; j < n; j++)
    {
      complex sum = 0.0;
      for (unsigned int i = 0; i < m; i++)
	sum += a[i][j] * tmp1[i];
      y[j] += sum;
    }
    
    // Evaluate third term
    complex tmp = 0.0;
    for (unsigned int k = 0; k < m; k++)
    {
      complex product = dot(a[k], x);
      for (unsigned int r = 0; r < m; r++)
	if ( r != k )
	  product *= tmp0[r];
      tmp += product;
    }
    for (unsigned int j = 1; j < n; j++)
    {
      complex sum = 0.0;
      for (unsigned int i = 0; i < m; i++)
	sum += w[i][j] * tmp;
      y[j] -= sum;
    }
  }

  unsigned int degree(unsigned int i) const
  {
    if ( i == 0 )
      return 1;
    else
      return m;
  }

};

#endif
