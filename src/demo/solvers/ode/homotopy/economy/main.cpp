// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

// Base class for economies

class Economy : public Homotopy
{
public:

  Economy(unsigned int m, unsigned int n) : Homotopy(n), m(m), n(n), a(0), w(0), tmp(0)
  {
    a = new real * [m];
    w = new real * [m];
    for (unsigned int i = 0; i < m; i++)
    {
      a[i] = new real[n];
      w[i] = new real[n];
      for (unsigned int j = 0; j < n; j++)
      {
	a[i][j] = dolfin::rand();
	w[i][j] = dolfin::rand();
      }
    }

    tmp = new complex[m];
    for (unsigned int i = 0; i < m; i++)
      tmp[i] = 0.0;
  }

  ~Economy()
  {
    for (unsigned int i = 0; i < m; i++)
    {
      delete [] a[i];
      delete [] w[i];
    }
    delete [] a;
    delete [] w;
    delete [] tmp;
  }

  void disp()
  {
    cout << "Utility parameters:" << endl;
    for (unsigned int i = 0; i < m; i++)
    {
      cout << "  trader " << i << ":";
      for (unsigned int j = 0; j < n; j++)
	cout << " " << a[i][j];
      cout << endl;
    }

    cout << "Initial endowments:" << endl;
    for (unsigned int i = 0; i < m; i++)
    {
      cout << "  trader " << i << ":";
      for (unsigned int j = 0; j < n; j++)
	cout << " " << w[i][j];
      cout << endl;
    }
  }

protected:

  // Compute sum of elements
  complex sum(const complex x[]) const
  {
    complex sum = 0.0;
    for (unsigned int j = 0; j < n; j++)
      sum += x[j];
    return sum;
  }

  // Compute scalar product x . y
  complex dot(const real x[], const complex y[]) const
  {
    complex sum = 0.0;
    for (unsigned int j = 0; j < n; j++)
      sum += x[j] * y[j];
    return sum;
  }

  // Display values
  void disp(const real x[], const char* name)
  {
    dolfin::cout << name << " = [";
    for (unsigned int j = 0; j < n; j++)
      dolfin::cout << x[j] << " ";
    dolfin::cout << "]" << endl;
  }

  // Display values
  void disp(const complex z[], const char* name)
  {
    dolfin::cout << name << " = [";
    for (unsigned int j = 0; j < n; j++)
      dolfin::cout << z[j] << " ";
    dolfin::cout << "]" << endl;
  }

  unsigned int m; // Number of traders
  unsigned int n; // Number of goods
  
  real** a; // Matrix of traders' preferences
  real** w; // Matrix of traders' initial endowments

  complex* tmp; // Temporary storage for scalar products
  
};

// Leontief economy

class Leontief : public Economy
{
public:

  Leontief(unsigned int m, unsigned int n) : Economy(m, n)
  {
    // Set condition that guarantees a solution for m = 1, n = 2
    /*
      if ( m == 1 && n == 2 )
      {
      dolfin_info("Setting condition for existence of solutions.");
      w[0][0] = a[0][0] * w[0][1] / a[0][1];
      }
    */
  }

  void F(const complex z[], complex y[])
  {
    // First equation: normalization
    y[0] = sum(z) - 1.0;

    // Precompute scalar products
    for (unsigned int i = 0; i < m; i++)
      tmp[i] = dot(w[i], z) / dot(a[i], z);

    // Evaluate right-hand side
    for (unsigned int j = 1; j < n; j++)
    {
      complex sum = 0.0;
      for (unsigned int i = 0; i < m; i++)
	sum += a[i][j] * tmp[i] - w[i][j];
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
      tmp[i] = (az*wx - wz*ax) / (az*az);
    }

    // Evaluate right-hand side
    for (unsigned int j = 1; j < n; j++)
    {
      complex sum = 0.0;
      for (unsigned int i = 0; i < m; i++)
	sum += a[i][j] * tmp[i];
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

// Constant elasticity of substitution (CES) economy

class CES : public Economy
{
public:

  CES(unsigned int m, unsigned int n, real epsilon) : Economy(m, n)
  {
    // Choose b such that epsilon < b_i < 1 for all i
    b = new real[m];
    for (unsigned int i = 0; i < m; i++)
      b[i] = epsilon + dolfin::rand()*(1.0 - epsilon);

    // Special choice of data (Eaves and Schmedders, two consumers)
    a[0][0] = 4.0; a[0][1] = 1.0;
    a[1][0] = 1.0; a[1][1] = 4.0;
    
    w[0][0] = 10.0; w[0][1] =  1.0;
    w[1][0] =  1.0; w[1][1] = 12.0;
    
    b[0] = 0.2;
    b[1] = 0.2;    

    // Special choice of data (Eaves and Schmedders, one consumer)
    /*
    a[0][0] = 4.0; a[0][1] = 1.0;
    a[1][0] = 1.0; a[1][1] = 4.0;
    
    w[0][0] = 10.0; w[0][1] =  1.0;
    w[1][0] =  0.0; w[1][1] =  0.0;
    
    b[0] = 0.2;
    b[1] = 0.2;
    */

    /*
      a[0][0] = 2.0; a[0][1] = 1.0;
      a[1][0] = 0.5; a[1][1] = 1.0;
      
      w[0][0] = 1.0; w[0][1] = 0.0;
      w[1][0] = 0.0; w[1][1] = 1.0;
      
      b[0] = 0.1;
      b[1] = 0.1;
    */
  }

  ~CES()
  {
    delete [] b;
  }

  /*
  complex z0(unsigned int i)
  {
    if ( i == 0 )
      return 1.01362052581566e-02;
    else
      return 9.89863794741843e-01;
  }
  */
  
  void F(const complex z[], complex y[])
  {
    // First equation: normalization
    y[0] = sum(z) - 1.0;
    
    // Precompute scalar products
    for (unsigned int i = 0; i < m; i++)
      tmp[i] = dot(w[i], z) / bdot(a[i], z, 1.0 - b[i]);
    
    // Evaluate right-hand side
    for (unsigned int j = 1; j < n; j++)
    {
      complex sum = 0.0;
      for (unsigned int i = 0; i < m; i++)
	sum += a[i][j] * std::pow(z[j], -b[i]) * tmp[i] - w[i][j];
      y[j] = sum;
    }
  }

  void JF(const complex z[], const complex x[], complex y[])
  {
    // First equation: normalization
    y[0] = sum(x);

    // First term
    for (unsigned int i = 0; i < m; i++)
    {
      const complex wx = dot(w[i], x);
      const complex az = bdot(a[i], z, 1.0 - b[i]);
      tmp[i] = wx / az;
    }
    for (unsigned int j = 1; j < n; j++)
    {
      complex sum = 0.0;
      for (unsigned int i = 0; i < m; i++)
	sum += a[i][j] * std::pow(z[j], -b[i]) * tmp[i];
      y[j] = sum;
    }

    // Second term
    for (unsigned int i = 0; i < m; i++)
    {
      const complex wz = dot(w[i], z);
      const complex az = bdot(a[i], z, 1.0 - b[i]);
      tmp[i] = b[i] * wz / az;
    }
    for (unsigned int j = 1; j < n; j++)
    {
      complex sum = 0.0;
      for (unsigned int i = 0; i < m; i++)
	sum += a[i][j] * std::pow(z[j], -1.0 - b[i]) * x[j] * tmp[i];
      y[j] -= sum;
    }

    // Third term
    for (unsigned int i = 0; i < m; i++)
    {
      const complex wz  = dot(w[i], z);
      const complex az  = bdot(a[i], z, 1.0 - b[i]);
      const complex axz = bdot(a[i], x, z, -b[i]);
      tmp[i] = (1.0 - b[i]) * wz * axz / (az * az);
    }
    for (unsigned int j = 1; j < n; j++)
    {
      complex sum = 0.0;
      for (unsigned int i = 0; i < m; i++)
	sum += a[i][j] * std::pow(z[j], -b[i]) * tmp[i];
      y[j] -= sum;
    }
  }

  /*

  void G(const complex z[], complex y[])
  {
    // First equation: normalization
    y[0] = sum(z) - 1.0;
    
    // Only one consumer
    const unsigned int i = 0;

    // Precompute scalar product
    tmp[i] = dot(w[i], z) / bdot(a[i], z, 1.0 - b[i]);
    
    // Evaluate right-hand side
    for (unsigned int j = 1; j < n; j++)
      y[j] = a[i][j] * std::pow(z[j], -b[i]) * tmp[i] - w[i][j];
  }

  void JG(const complex z[], const complex x[], complex y[])
  {
    // First equation: normalization
    y[0] = sum(x);

    // Only one consumer
    const unsigned int i = 0;

    // First term
    complex wx = dot(w[i], x);
    complex az = bdot(a[i], z, 1.0 - b[i]);
    tmp[i] = wx / az;
    for (unsigned int j = 1; j < n; j++)
      y[j] = a[i][j] * std::pow(z[j], -b[i]) * tmp[i];

    // Second term
    complex wz = dot(w[i], z);
    az = bdot(a[i], z, 1.0 - b[i]);
    tmp[i] = b[i] * wz / az;
    for (unsigned int j = 1; j < n; j++)
	y[j] -= a[i][j] * std::pow(z[j], -1.0 - b[i]) * x[j] * tmp[i];
    
    // Third term
    wz  = dot(w[i], z);
    az  = bdot(a[i], z, 1.0 - b[i]);
    complex axz = bdot(a[i], x, z, -b[i]);
    tmp[i] = (1.0 - b[i]) * wz * axz / (az * az);
    for (unsigned int j = 1; j < n; j++)
      y[j] = a[i][j] * std::pow(z[j], -b[i]) * tmp[i];
  }

  */

  unsigned int degree(unsigned int i) const
  {
    if ( i == 0 )
      return 1;
    else
      return m;
  }
  
private:

  // Compute special scalar product x . y.^b
  complex bdot(const real x[], const complex y[], real b) const
  {
    complex sum = 0.0;
    for (unsigned int j = 0; j < n; j++)
      sum += x[j] * std::pow(y[j], b);
    return sum;
  }

  // Compute special scalar product x . y. z^b
  complex bdot(const real x[], const complex y[], const complex z[], real b) const
  {
    complex sum = 0.0;
    for (unsigned int j = 0; j < n; j++)
      sum += x[j] * y[j] * std::pow(z[j], b);
    return sum;
  }

  real* b; // Vector of exponents

};

// Economy with two goods from Polemarchakis's "On the transfer paradox"

class Polemarchakis : public Homotopy
{
public:

  Polemarchakis() : Homotopy(2), lambda(0), gamma(0), polynomial(true)
  {
    lambda = new real[3];
    gamma = new real[3];

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
    if ( polynomial )
    {
      y[0] = ( gamma[0]*(z[0] + lambda[1])*(z[0] + lambda[2]) +
	       gamma[1]*(z[0] + lambda[0])*(z[0] + lambda[2]) +
	       gamma[2]*(z[0] + lambda[0])*(z[0] + lambda[1]) );
    }
    else
    {
      complex sum = 0.0;
      for (unsigned int i = 0; i < 3; i++)
	sum += gamma[i] / (z[0] + lambda[i]);
      y[0] = sum;
    }

    // Second equation
    y[1] = z[1] - 1.0;
  }

  void JF(const complex z[], const complex x[], complex y[])
  {
    // First equation
    if ( polynomial )
    {
      y[0] = ( gamma[0]*(2.0*z[0] + lambda[1] + lambda[2]) + 
	       gamma[1]*(2.0*z[0] + lambda[0] + lambda[2]) + 
	       gamma[2]*(2.0*z[0] + lambda[0] + lambda[1]) ) * x[0];
    }
    else
    {
      complex sum = 0.0;
      for (unsigned int i = 0; i < 3; i++)
      {
	const complex tmp = z[0] + lambda[i];
	sum -= gamma[i] / (tmp*tmp);
      }
      y[0] = sum * x[0];
    }
    
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

  real* lambda;
  real* gamma;

  bool polynomial;

};

int main()
{
  dolfin_set("method", "cg");
  dolfin_set("order", 1);
  dolfin_set("adaptive samples", true);
  //dolfin_set("homotopy monitoring", true);
  dolfin_set("tolerance", 0.01);
  dolfin_set("initial time step", 0.01);
  dolfin_set("homotopy divergence tolerance", 10.0);
  dolfin_set("homotopy randomize", true);
  dolfin_set("linear solver", "direct");

  //Leontief leontief(2, 2);
  //leontief.solve();

  //CES ces(2, 2, 0.5);
  //ces.disp();
  //ces.solve();

  Polemarchakis polemarchakis;
  polemarchakis.solve();

  return 0;
}
