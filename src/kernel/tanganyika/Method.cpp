// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Method.h>

//-----------------------------------------------------------------------------
Method::Method(int iiHighestOrder)
{
  assert ( iiHighestOrder >= 0 );

  // Set bOK to true
  bOK = true;
  
  // Set the order
  iHighestOrder = iiHighestOrder;
  
  // Allocate memory for the nodal points
  dNodalPoints = new double*[iiHighestOrder+1];
  for (int i=0;i<(iiHighestOrder+1);i++)
	 dNodalPoints[i] = new double[i+1];
  
  // Allocate memory for the basis functions
  lBasisFunctions = new Lagrange**[iiHighestOrder+1];
  for (int i=0;i<(iiHighestOrder+1);i++){
  	 lBasisFunctions[i] = new Lagrange*[i+1];
	 for (int j=0;j<(i+1);j++)
		lBasisFunctions[i][j] = new Lagrange(i);
  }
	 
  // Allocate memory for the weights
  dNodalWeights = new double**[iHighestOrder+1];
  for (int i=0;i<(iHighestOrder+1);i++){
	 dNodalWeights[i] = new double*[i+1];
	 for (int j=0;j<(i+1);j++){
		dNodalWeights[i][j] = new double[i+1];
	 }
  }

  // Allocate memory for the weight function nodal values
  dWeightFunctionNodalValues = new double**[iHighestOrder+1];
  for (int i=0;i<(iHighestOrder+1);i++){
	 dWeightFunctionNodalValues[i] = new double*[i+1];
	 for (int j=0;j<(i+1);j++){
		dWeightFunctionNodalValues[i][j] = new double[i+1];
	 }
  }
  
  // Allocate memory for the derivative weights
  dDerivativeWeights = new double*[iHighestOrder+1];
  for (int i=0;i<(iHighestOrder+1);i++)
	 dDerivativeWeights[i] = new double[i+1];

  // Allocate memory for the stability weights
  dStabilityWeights = new double*[iHighestOrder+1];
  for (int i=0;i<(iHighestOrder+1);i++)
	 dStabilityWeights[i] = new double[i+1];

  // Allocate memory for the interpolation constants
  dInterpolationConstants = new double[iHighestOrder+2];

  // Allocate memory for the residual factors
  dResidualFactors = new double[iHighestOrder+1];

  // Allocate memory for the product factors
  dProductFactors = new double[iHighestOrder+1];

  // Allocate memory for the quadrature factors
  dQuadratureFactors = new double[iHighestOrder+1];

  // Allocate memory for the interpolation points
  dInterpolationPoints = new double*[iHighestOrder+1];
  for (int i=0;i<(iHighestOrder+1);i++)
    dInterpolationPoints[i] = new double[i+1];
  
  // Allocate memory for the interpolation weights
  dInterpolationWeights = new double*[iHighestOrder+1];
  for (int i=0;i<(iHighestOrder+1);i++)
	 dInterpolationWeights[i] = new double[i+1];

  // Initialize the linearizations
  // Needs to be done individually

  // Initialize indicator for LU factorization
  bLinearizationSingular = new bool[iHighestOrder+1];
  for (int i=0;i<(iHighestOrder+1);i++)
	 bLinearizationSingular[i] = false;
  
}
//-----------------------------------------------------------------------------
Method::~Method()
{
  // Delete the nodal points
  for (int i=0;i<(iHighestOrder+1);i++)
    delete dNodalPoints[i];
  delete dNodalPoints;
  
  // Delete the basis functions
  for (int i=0;i<(iHighestOrder+1);i++){
    for (int j=0;j<(i+1);j++)
      delete lBasisFunctions[i][j];
    delete lBasisFunctions[i];
  }
  delete lBasisFunctions;
  
  // Delete the nodal weights
  for (int i=0;i<(iHighestOrder+1);i++){
	 for (int j=0;j<(i+1);j++)
		delete dNodalWeights[i][j];
	 delete dNodalWeights[i];
  }
  delete dNodalWeights;
  
  // Delete the weight function nodal values
  for (int i=0;i<(iHighestOrder+1);i++){
	 for (int j=0;j<(i+1);j++)
		delete dWeightFunctionNodalValues[i][j];
	 delete dWeightFunctionNodalValues[i];
  }
  delete dWeightFunctionNodalValues;

  // Delete the derivative weights
  for (int i=0;i<(iHighestOrder+1);i++)
	 delete dDerivativeWeights[i];
  delete dDerivativeWeights;

  // Delete the stability weights
  for (int i=0;i<(iHighestOrder+1);i++)
	 delete dStabilityWeights[i];
  delete dStabilityWeights;

  // Delete the interpolation constants
  delete dInterpolationConstants;

  // Delete the residual factors
  delete dResidualFactors;
  
  // Delete the product factors
  delete dProductFactors;

  // Delete the quadrature factors
  delete dQuadratureFactors;
  
  // Delete the interpolation points
  for (int i=0;i<(iHighestOrder+1);i++)
    delete dInterpolationPoints[i];
  delete dInterpolationPoints;
  
  // Delete the interpolation weights
  for (int i=0;i<(iHighestOrder+1);i++)
    delete dInterpolationWeights[i];
  delete dInterpolationWeights;
  
  // Delete indicator for LU factorization
  delete bLinearizationSingular;
  
}
//-----------------------------------------------------------------------------
  void Method::Init()
  {
  // This function initializes the method.

  // Initialize the quadrature
  InitQuadrature();
  
  // Get the nodal points
  GetNodalPoints();
  
  // Compute the nodal basis functions
  ComputeNodalBasis();
  
  // Compute the nodal weights
  ComputeNodalWeights();
	 
  // Compute derivative weights
  ComputeDerivativeWeights();
	 
  // Compute interpolation constants
  ComputeInterpolationConstants();

  // Compute residual factors
  ComputeResidualFactors();
  
  // Compute product factors
  ComputeProductFactors();
  
  // Compute quadrature factors
  ComputeQuadratureFactors();

  // Compute interpolation weights
  ComputeInterpolationWeights();
}
//-----------------------------------------------------------------------------
double Method::GetNodalPoint(int iOrder, int iNodalIndex)
{
  // This function returns the normalized position in the interval [0,1]
  // of the iNodalIndex:th node for elements of order iOrder.

  assert ( iOrder >= 0 );
  assert ( iOrder <= iHighestOrder );
  assert ( iNodalIndex >= 0 );
  assert ( iNodalIndex <= iOrder );
  
  return ( dNodalPoints[iOrder][iNodalIndex] );
}
//-----------------------------------------------------------------------------
double Method::GetBasisFunction(int iOrder, int iIndex, double dTau)
{
  // This function returns the value of the iIndex:th basis function
  // for elements of order iOrder at dTau, where dTau is the
  // normalized time value in the interval [0,1].

  assert ( iOrder >= 0 );
  assert ( iOrder <= iHighestOrder );
  assert ( iIndex >= 0 );
  assert ( iIndex <= iOrder );

  return ( lBasisFunctions[iOrder][iIndex]->Value(dTau) );
}
//-----------------------------------------------------------------------------
double Method::GetDerivativeRight(int iOrder, int iIndex)
{
  // This function returns the value of the derivative of the iIndex:th
  // basis function for elements of order iOrder at the right end-point
  // of the interval.

  assert ( iOrder >= 0 );
  assert ( iOrder <= iHighestOrder );
  assert ( iIndex >= 0 );
  assert ( iIndex <= iOrder );
  
  return ( dDerivativeWeights[iOrder][iIndex] );
}
//-----------------------------------------------------------------------------
double Method::GetStabilityWeight(int iOrder, int iIndex)
{
  // This function returns the value of the q:th derivative of the iIndex:th
  // basis function for elements of order iOrder.

  assert ( iOrder >= 0 );
  assert ( iOrder <= iHighestOrder );
  assert ( iIndex >= 0 );
  assert ( iIndex <= iOrder );
  
  return ( dStabilityWeights[iOrder][iIndex] );
}
//-----------------------------------------------------------------------------
double Method::GetWeight(int iOrder, int iIndex, int iNodalPoint)
{
  // This function returns the Galerkin quadrature weight for nodalpoint
  // iNodalPoint for  elements of order iOrder, used for computing
  // integral iIndex.
  //
  // Note! iNodalPoint == 0 gives 0.0 for cG(q) elements. This index
  // is not used anyway for cG(q) elements.

  assert ( iOrder >= 0 );
  assert ( iOrder <= iHighestOrder );
  assert ( iIndex >= 0 );
  assert ( iIndex <= iOrder );
  assert ( iNodalPoint >= 0 );
  assert ( iNodalPoint <= iOrder );
  
  return ( dNodalWeights[iOrder][iIndex][iNodalPoint] );
}
//-----------------------------------------------------------------------------
double Method::GetInterpolationConstant(int iOrder)
{
  // This function returns the interpolation constant

  assert ( iOrder >= 0 );
  assert ( iOrder <= (iHighestOrder+1) );

  // We might want to use C_(q+1) - second estimate with tilde
  
  return ( dInterpolationConstants[iOrder] );
}
//-----------------------------------------------------------------------------
double Method::GetResidualFactor(int iOrder)
{
  // This function returns the residual factor

  assert ( iOrder >= 0 );
  assert ( iOrder <= iHighestOrder );

  return ( dResidualFactors[iOrder] );
}
//-----------------------------------------------------------------------------
double Method::GetProductFactor(int iOrder)
{
  // This function returns the product factor
  
  assert ( iOrder >= 0 );
  assert ( iOrder <= iHighestOrder );

  return ( dProductFactors[iOrder] );
}
//-----------------------------------------------------------------------------
double Method::GetQuadratureFactor(int iOrder)
{
  // This function returns the quadrature factor, i.e.
  //
  // E_i / (E_i - E_{i+1})
  
  assert ( iOrder >= 0 );
  assert ( iOrder <= iHighestOrder );

  return ( dQuadratureFactors[iOrder] );
}
//-----------------------------------------------------------------------------
double Method::GetInterpolationPoint(int iOrder, int iNodalIndex)
{
  // This function returns the interpolation point.

  assert ( iOrder      >= 0 );
  assert ( iOrder      <= iHighestOrder );
  assert ( iNodalIndex >= 0 );
  assert ( iNodalIndex <= iOrder );

  return ( dInterpolationPoints[iOrder][iNodalIndex] );
}
//-----------------------------------------------------------------------------
double Method::GetInterpolationWeight(int iOrder, int iNodalIndex)
{
  // This function returns the interpolation weight

  assert ( iOrder      >= 0 );
  assert ( iOrder      <= iHighestOrder );
  assert ( iNodalIndex >= 0 );
  assert ( iNodalIndex <= iOrder );

  return ( dInterpolationWeights[iOrder][iNodalIndex] );
}
//-----------------------------------------------------------------------------
bool Method::DataOK()
{
  // This function returns true if the computation of nodal points and
  // weight went ok, false otherwise.

  return ( bOK );
}
//-----------------------------------------------------------------------------
void Method::CheckWeight(double dWeight)
{
  // This function checks that the weight is correct. It should be 1.

  if ( fabs(dWeight-1.0) > DEFAULT_PRECALC_CHECK_TOL )
	 bOK = false;  
}
//-----------------------------------------------------------------------------
void Method::ComputeNodalBasis()
{
  // This function computes the nodal basis, i.e. computes the lagrange
  // polynomials from the nodal points.

  double c;
  int    iPosition;

  // Set the polynomials for order 0
  lBasisFunctions[0][0]->SetConstant(1.0);
  
  // Compute the polynomials for every order greater than 1
  for (int i=1;i<(iHighestOrder+1);i++)
	 for (int j=0;j<(i+1);j++){
		
		// Compute polynomial j for order (i+1)
		
		// First compute the constant
		c = 1.0;
		for (int k=0;k<(i+1);k++)
		  if ( k != j )
			 c *= ( dNodalPoints[i][j] - dNodalPoints[i][k] );
		c = 1/c;
		
		// Set coefficients for Lagrange basis functions
		iPosition = 0;
		for (int k=0;k<(i+1);k++)
		  if ( k != j )
			 lBasisFunctions[i][j]->SetPoint(iPosition++,dNodalPoints[i][k]);
		lBasisFunctions[i][j]->SetConstant(c);
		
	 }

}
//-----------------------------------------------------------------------------
void Method::ComputeDerivativeWeights()
{
  // This function computes constants for computing
  //
  //         1. The derivative at the end point
  //         2. The q:th order derivative which is a constant

  // Funkar kanske inte för dG?
  // FIXME
  
  
  // Fix q = 0
  dDerivativeWeights[0][0] = 0.0;
  dStabilityWeights[0][0]  = 0.0;
  
  // Compute weights for the different orders
  for (int i=1;i<(iHighestOrder+1);i++){
	 
	 // Compute the derivative weights
	 for (int j=0;j<(i+1);j++)
		dDerivativeWeights[i][j] = lBasisFunctions[i][j]->Derivative(1.0);
	 
	 // Compute the stability weights
	 for (int j=0;j<(i+1);j++)
		dStabilityWeights[i][j] = lBasisFunctions[i][j]->NthDerivative();
	 
  }
  
}
//-----------------------------------------------------------------------------
double Method::Value(double *dVals, int iOrder, double dTau)
{
  // This function computes the value of the polynomial of degree iOrder
  // defined by dVals at dTau.

  double dValue = 0.0;

  for (int i=0;i<=iOrder;i++)
	 dValue += lBasisFunctions[iOrder][i]->Value(dTau) * dVals[i];

  return ( dValue );
}
//-----------------------------------------------------------------------------
