// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

class Leontief : public Homotopy
{
public:

  Leontief(unsigned int m, unsigned int n) : Homotopy(n), m(m), n(n), tmp(0)
  {
    // Initialize temporary storage
    tmp = new complex[m];
    for (unsigned int i = 0; i < m; i++)
      tmp[i] = 0.0;

    // Initialize matrices a and w
    a = new real * [m];
    w = new real * [m];
    for (unsigned int i = 0; i < m; i++)
    {
      a[i] = new real[n];
      w[i] = new real[n];
    }
    
    // Randomize a and w
    for (unsigned int i = 0; i < m; i++)
    {
      for (unsigned int j = 0; j < n; j++)
      {
	a[i][j] = dolfin::rand();
	w[i][j] = dolfin::rand();
      }
    }
  }

  ~Leontief()
  {
    if ( tmp ) delete [] tmp;
    for (unsigned i = 0; i < m; i++)
    {
      delete [] a[i];
      delete [] w[i];
    }
    delete [] a;
    delete [] w;
  }

  void F(const complex z[], complex y[])
  {
    for (unsigned int j = 0; j < n; j++)
      y[j] = 0.0;

    // Precompute scalar products
    for (unsigned int i = 0; i < m; i++)
      tmp[i] = dot(w[i], z) / dot(a[i], z);

    // Evaluate right-hand side
    for (unsigned int j = 0; j < n; j++)
    {
      complex sum = 0.0;
      for (unsigned int i = 0; i < m; i++)
	sum += a[i][j] * tmp[i] - w[i][j];
      y[j] = sum;
    }
  }

  void JF(const complex z[], const complex x[], complex y[])
  {
    // Precompute scalar products
    for (unsigned int i = 0; i < m; i++)
    {
      const complex az = dot(a[i], z);
      const complex wz = dot(w[i], z);
      const complex ax = dot(a[i], x);
      const complex wx = dot(w[i], x);
      tmp[i] = (az*wx - wz*ax) / (az*az);
    }

    // Evaluate right-hand side
    for (unsigned int j = 0; j < n; j++)
    {
      complex sum = 0.0;
      for (unsigned int i = 0; i < m; i++)
	sum += a[i][j] * tmp[i];
      y[j] = sum;
    }
  }

  unsigned int degree(unsigned int i) const
  {
    return m;
  }
  
private:

  complex dot(const real x[], const complex y[]) const
  {
    complex sum = 0.0;
    for (unsigned int i = 0; i < n; i++)
      sum += x[i] * y[i];
    return sum;
  }

  unsigned int m; // Number of traders
  unsigned int n; // Number of goods

  real** a;     // Matrix of traders' preferences
  real** w;     // Matrix of traders' initial endowments
  complex* tmp; // Temporary storage for scalar products

};

int main()
{
  dolfin_set("method", "cg");
  dolfin_set("order", 2);

  Leontief leontief(2, 1);
  leontief.solve();

  return 0;
}
