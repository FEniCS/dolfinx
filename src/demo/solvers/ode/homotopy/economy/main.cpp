// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

class Economy : public Homotopy
{
public:

  Economy(unsigned int m, unsigned int n) : Homotopy(n), m(m), n(n), a(0), w(0)
  {
    // Initialize matrices a and w
    a = new real * [m];
    w = new real * [m];
    for (unsigned int i = 0; i < m; i++)
    {
      a[i] = new real[n];
      w[i] = new real[n];
      for (unsigned int j = 0; j < n; j++)
      {
	a[i][j] = 0.0;
	w[i][j] = 0.0;
      }
    }
  }

  ~Economy()
  {
    for (unsigned i = 0; i < m; i++)
    {
      delete [] a[i];
      delete [] w[i];
    }
    delete [] a;
    delete [] w;
  }

protected:

  unsigned int m; // Number of traders
  unsigned int n; // Number of goods
  
  real** a; // Matrix of traders' preferences
  real** w; // Matrix of traders' initial endowments
  
};

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

    // Set condition that guarantees a solution for m = 1, n = 2
    /*
      if ( m == 1 && n == 2 )
      {
      dolfin_info("Setting condition for existence of solutions.");
      w[0][0] = a[0][0] * w[0][1] / a[0][1];
      }
    */
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
  
  real** a; // Matrix of traders' preferences
  real** w; // Matrix of traders' initial endowments

  complex* tmp; // Temporary storage for scalar products

};

int main()
{
  dolfin_set("method", "cg");
  dolfin_set("order", 1);
  dolfin_set("adaptive samples", true);
  dolfin_set("tolerance", 0.05);

  Leontief leontief(1, 2);
  leontief.solve();

  return 0;
}
