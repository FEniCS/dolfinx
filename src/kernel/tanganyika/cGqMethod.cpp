// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/cGqMethod.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
cGqMethod::cGqMethod(int q) : Method(int q)
{
  if ( q < 1 )
    dolfin_error("Polynomial order q must be at least 1 for the cG(q) method.");

  init();
}
//-----------------------------------------------------------------------------
void cGqMethod::computeQuadrature()
{
  // Use Lobatto quadrature
  Lobatto quadrature(q);

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
  Lobatto lobatto(q-1);
  for (int i = 0; i < (n-1); i++)
    test->set(i, (lobatto->point(i) + 1.0) / 2.0);
}
//-----------------------------------------------------------------------------
void cGqMethod::computeWeights()
{
  Matrix A(n, n, Matrix::DENSE);
  Matrix B(n, n, Matrix::DENSE);
  
  // Compute matrix coefficients
  for (int i = 1; i <= q; i++) {
    for (int j = 1; j <= q; j++) {
      
      // Use Lobatto quadrature which is exact for the order we need, 2q-1
      real integral = 0.0;
      for (int k = 0; k < n; k++) {
	x = points[k];
	integral += qweight[k] * trial->dx(j,x) * test->(i-1,x);
      }
      
      A(i-1,j-1) = integral;
      
    }
  }
  
  // Compute inverse
  A.hpinverse(B);
  
  // Compute nodal weights
  for (int j=1;j<(i+1);j++) {
    
    for (int k=0;k<(i+1);k++) {
      
      // Order:            i
      // Integral:         j (=1,...,i)
      // Quadrature point: k (=0,...,i)
      
      // Compute the sum
      dWeight = 0.0;
      for (int l=0;l<i;l++)
	dWeight += (B->Get(j-1,l)) * (lBasisFunctions[i-1][l]->Value(dNodalPoints[i][k]));
      
      // Check that the weight is correct. All weights for the last integral, j = i,
      // should be 1.
      if ( j == i )
	CheckWeight(dWeight);
      
      // Use the quadrature weight
      dWeight *= (qQuadrature->GetWeight(i+1,k));
      
      // Compensate for integrating over [0,1] and not [-1,1]
      dWeight *= 0.5;
      
      // Set the weight
      dNodalWeights[i][j][k] = dWeight;
      
    }
    
  }
  
  // Save the weight function
  for (int i = 1; i < (i+1); j++)
    for (int k = 0; k < i; k++)
      dWeightFunctionNodalValues[i][j][k] = B->Get(j-1,k);
  
  // Set the weight for j=0 which is not used for cG(q)
  for (int k=0;k<=i;k++)
    dNodalWeights[i][0][k] = 0.0;
  

}
//-----------------------------------------------------------------------------
void cGqMethod::Display()
{
  // This function displays the details of the method

  cout << "Data for the cG(q) method, q = 1,...," << iHighestOrder << ":" << endl;
  cout << "========================================" << endl << endl;
  
  // Display quadrature
  cout << "Quadrature for nodal basis: " << endl << endl;
  qQuadrature->Display();

  cout << endl;
  cout << "Quadrature for interpolation: " << endl << endl;
  qGauss->Display();

  // Display quadrature
  qQuadrature->Display();
  
  // Display basis functions
  cout << endl;
  for (int i=1;i<(iHighestOrder+1);i++){
	 cout << "Lagrange basis functions for order q = " << i << ":" << endl << endl;
	 for (int j=0;j<(i+1);j++)
		cout << "  " << *lBasisFunctions[i][j] << endl; 
	 cout << endl;
  }
  
  // Display the weights
  cout << endl;
  for (int i=1;i<(iHighestOrder+1);i++){
	 cout << "Nodal weights for order q = " << i << ":" << endl << endl;
	 for (int j=1;j<(i+1);j++){
		cout << "  k = " << j+1 << ": [ ";
		for (int k=0;k<(i+1);k++)
		  cout << dNodalWeights[i][j][k] << " "; 
		cout << "]" << endl;
		
	 }
	 cout << endl;
  }

  // Display the weight functions
  cout << endl;
  for (int i=1;i<(iHighestOrder+1);i++){
	 cout << "Weight functions for order q = " << i << ":" << endl << endl;
	 for (int j=1;j<(i+1);j++){
		cout << "  k = " << j+1 << ": [ ";
		for (int k=0;k<i;k++)
		  cout << dWeightFunctionNodalValues[i][j][k] << " "; 
		cout << "]" << endl;
		
	 }
	 cout << endl;
  }

  // Display the derivative weights
  cout << endl;
  for (int i=1;i<(iHighestOrder+1);i++){
	 cout << "Derivative weights at right endpoint for order q = " << i << ":";
	 cout << endl << endl;
	 cout << "  [ ";
	 for (int j=0;j<(i+1);j++)
		cout << dDerivativeWeights[i][j] << " "; 
	 cout << "]" << endl;
	 cout << endl;
  }
  cout << endl;

  // Display the stability weights
  cout << endl;
  for (int i=1;i<(iHighestOrder+1);i++){
	 cout << "q:th derivative weights for order q = " << i << ":";
	 cout << endl << endl;
	 cout << "  [ ";
	 for (int j=0;j<(i+1);j++)
		cout << dStabilityWeights[i][j] << " "; 
	 cout << "]" << endl;
	 cout << endl;
  }
  cout << endl;

  // Display the interpolation constants
  cout << "Interpolation constants:" << endl << endl;
  for (int i=1;i<=(iHighestOrder+1);i++)
	 cout << "  C_" << i << " = " << dInterpolationConstants[i] << endl;
  cout << endl;


  // Display the residual factors
  cout << "Residual factors:" << endl << endl;
  for (int i=1;i<(iHighestOrder+1);i++)
	 cout << "  C_" << i << " = " << dResidualFactors[i] << endl;
  cout << endl;

  // Display the product factors
  cout << "Product factors:" << endl << endl;
  for (int i=1;i<(iHighestOrder+1);i++)
	 cout << "  C_" << i << " = " << dProductFactors[i] << endl;
  cout << endl;

  // Display the quadrature factors
  cout << "Quadrature factors:" << endl << endl;
  for (int i=1;i<(iHighestOrder+1);i++)
	 cout << "  C_" << i << " = " << dQuadratureFactors[i] << endl;
  cout << endl;

  // Display the interpolation weights
  cout << endl;
  for (int i=1;i<(iHighestOrder+1);i++){
	 cout << "Interpolation weights for order q = " << i << ":";
	 cout << endl << endl;
	 cout << "  [ ";
	 for (int j=0;j<i;j++)
		cout << dInterpolationWeights[i][j] << " "; 
	 cout << "]" << endl;
	 cout << endl;
  }
  cout << endl;

  // Display the interpolation points
  cout << endl;
  for (int i=1;i<(iHighestOrder+1);i++){
	 cout << "Interpolation points for order q = " << i << ":";
	 cout << endl << endl;
	 cout << "  [ ";
	 for (int j=0;j<i;j++)
		cout << dInterpolationPoints[i][j] << " "; 
	 cout << "]" << endl;
	 cout << endl;
  }
  cout << endl;
}
