// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Lagrange.h>
#include <dolfin/LobattoQuadrature.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/cGqMethod.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
cGqMethod::cGqMethod(int q) : Method(q)
{
  if ( q < 1 )
    dolfin_error("Polynomial order q must be at least 1 for the cG(q) method.");

  init();
}
//-----------------------------------------------------------------------------
void cGqMethod::show() const
{
  dolfin_info("Data for the cG(%d) method", q);
  dolfin_info("==========================");
  dolfin_info("");

  dolfin_info("Lobatto quadrature points and weights on [0,1]:");
  dolfin_info("");
  dolfin_info(" i   points                   weights");
  dolfin_info("----------------------------------------------------");
  
  for (int i = 0; i < n; i++)
    dolfin_info("%2d   %.16e   %.16e", i, points[i], qweights[i]);

  for (int i = 1; i < n; i++) {
    dolfin_info("");
    dolfin_info("cG(%d) weights for degree of freedom %d:", q, i);
    dolfin_info("");
    dolfin_info(" i   weights");
    dolfin_info("---------------------------");
    for (int j = 0; j < n; j++)
      dolfin_info("%2d   %.16e", j, weights[i][j]);
  }
}
//-----------------------------------------------------------------------------
void cGqMethod::computeQuadrature()
{
  // Use Lobatto quadrature
  LobattoQuadrature quadrature(n);

  // Get points and rescale from [-1,1] to [0,1]
  for (int i = 0; i < n; i++)
    points[i] = (quadrature.point(i) + 1.0) / 2.0;

  // Get weights and rescale from [-1,1] to [0,1]
  for (int i = 0; i < n; i++)
    qweights[i] = 0.5 * quadrature.weight(i);
}
//-----------------------------------------------------------------------------
void cGqMethod::computeBasis()
{
  dolfin_assert(!trial);
  dolfin_assert(!test);

  // Compute Lagrange basis for trial space
  trial = new Lagrange(q);
  for (int i = 0; i < n; i++)
    trial->set(i, points[i]);

  // Compute Lagrange basis for test space using the Lobatto points for q-1
  test = new Lagrange(q-1);
  if ( q > 1 ) {
    LobattoQuadrature lobatto(n-1);
    for (int i = 0; i < (n-1); i++)
      test->set(i, (lobatto.point(i) + 1.0) / 2.0);
  }
  else
    test->set(0, 1.0);
}
//-----------------------------------------------------------------------------
void cGqMethod::computeWeights()
{
  Matrix A(q, q, Matrix::dense);
  
  // Compute matrix coefficients
  for (int i = 1; i < n; i++) {
    for (int j = 1; j < n; j++) {
  
      // Use Lobatto quadrature which is exact for the order we need, 2q-1
      real integral = 0.0;
      for (int k = 0; k < n; k++) {
	real x = points[k];
	integral += qweights[k] * trial->dx(j,x) * test->eval(i-1,x);
      }
      
      A(i-1,j-1) = integral;
      
    }
  }

  Vector b(q);
  Vector w(q);

  // Compute nodal weights for each degree of freedom (loop over points)
  for (int i = 0; i < n; i++) {
    
    // Get nodal point
    real x = points[i];
    
    // Evaluate test functions at current nodal point
    for (int j = 0; j < q; j++)
      b(j) = test->eval(j,x);
    
    // Solve for the weight functions at the nodal point
    A.hpsolve(w,b);

    // Save weights including quadrature
    for (int j = 1; j < n; j++)
      weights[j][i] = qweights[i] * w(j-1);
    
  }

  // Set weights for i = 0 (not used)
  for (int j = 0; j < n; j++)
    weights[0][j] = 0.0;
}
//-----------------------------------------------------------------------------
