// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/cGqMethod.h>

using namespace dolfin;

real dPrecalcResidualFactors_cG[PRECALC_MAX+1] = PRECALC_CG_RF;
real dPrecalcProductFactors_cG[PRECALC_MAX+1] = PRECALC_CG_PF;

//-----------------------------------------------------------------------------
    cGqMethod(int iiHighestOrder) : Method(iiHighestOrder){
      // Initialize linearizations
      mLinearizations = new Matrix<real>* [iiHighestOrder+1];
      mSteps          = new Matrix<real>* [iiHighestOrder+1];
      mLinearizations[0] = new Matrix<real>(1,1);
      mSteps[0]          = new Matrix<real>(1,1);
      for (int i=1;i<(iiHighestOrder+1);i++){
	mLinearizations[i] = new Matrix<real>(i,i);
	mSteps[i]          = new Matrix<real>(i,1);	 
      }
    };	 


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
//-----------------------------------------------------------------------------
real cGqMethod::GetWeightGeneric(int iOrder, int iIndex, real dTau)
{
  // This function returns the Galerkin quadrature weight at dTau
  // for  elements of order iOrder, used for computing integral iIndex.
  //

  // Note! This value does not include the quadrature weight!
    
  assert ( iOrder >= 1 );
  assert ( iOrder <= iHighestOrder );
  assert ( iIndex >= 1 );
  assert ( iIndex <= iOrder );

  real dValue = 0.0;

  // Compute the value of the basis function
  
  for (int i=0;i<iOrder;i++)
	 dValue += ( lBasisFunctions[iOrder-1][i]->Value(dTau) *
					 dWeightFunctionNodalValues[iOrder][iIndex][i] );
  
  return ( dValue );
}
//-----------------------------------------------------------------------------
real cGqMethod::GetQuadratureWeight(int iOrder, int iNodalIndex)
{
  // This function returns the quadrature weight for node iNodalIndex for
  // elements of order iOrder.
  
  assert ( iOrder >= 1 );
  assert ( iOrder <= iHighestOrder );
  assert ( iNodalIndex >= 0 );
  assert ( iNodalIndex <= iOrder );
  
  return ( (qQuadrature->GetWeight(iOrder+1,iNodalIndex))/2.0 );
}
//-----------------------------------------------------------------------------
void cGqMethod::UpdateLinearization(int    iOrder,
												real dTimeStep,
												real dDerivative)
{
  // This function updates the linearization.
  
  // Note! This may not work very well for generic quadrature.
  // It is assumed that the quadrature points and nodal points
  // are the same.

  real dVal;
  real kdfdu = dTimeStep * dDerivative;

  // Compute the matrix coefficients
  for (int i=0;i<iOrder;i++)
	 for (int j=0;j<iOrder;j++){

		// Compute the value
		if ( i == j )
		  dVal = 1.0;
		else
		  dVal = 0.0;
		dVal -= kdfdu * dNodalWeights[iOrder][i+1][j+1];

		// Set the value in the matrix
		mLinearizations[iOrder]->Set(i,j,dVal);
		
	 }

  // Compute the LU factorization
  if ( !(mLinearizations[iOrder]->LU()) )
	 bLinearizationSingular[iOrder] = true;
  else
	 bLinearizationSingular[iOrder] = false;  
  
}
//-----------------------------------------------------------------------------
void cGqMethod::GetStep(int iOrder, real *dStep)
{
  // This function computes the new step using the linearization.

  // If the matrix was singular then do fix point interation
  if ( bLinearizationSingular[iOrder] )
	 return;
  
  // Set the right-hand side coefficients
  for (int i=0;i<iOrder;i++)
	 mSteps[iOrder]->Set(i,0,dStep[i]);
  
  // Solve
  mLinearizations[iOrder]->LUSolve(mSteps[iOrder]);
  
  // Place the solution in dStep
  for (int i=0;i<iOrder;i++)
	 dStep[i] = mSteps[iOrder]->Get(i,0);
  
}
//-----------------------------------------------------------------------------
void cGqMethod::InitQuadrature()
{
  // This function initializes the quadrature.
  
  qQuadrature = new Lobatto(iHighestOrder+1);
  if ( !(qQuadrature->Init()) )
	 bOK = false;

  qGauss = new Gauss(iHighestOrder+1);
  if ( !(qGauss->Init()) )
	 bOK = false;

}
//-----------------------------------------------------------------------------
void cGqMethod::GetNodalPoints()
{
  // This function computes the nodal points for basis functions
  // and quadrature.

  // Get the points and recompute to [0,1]
  for (int i=0;i<(iHighestOrder+1);i++)
	 for (int j=0;j<(i+1);j++)
		dNodalPoints[i][j] = (qQuadrature->GetPoint(i+1,j) + 1.0) / 2.0; 
}
//-----------------------------------------------------------------------------
void cGqMethod::ComputeNodalWeights()
{
  // This function computes the nodal weights for the cG(q) method.

  Matrix<real> *A, *B;
  Polynomial<real> p1(0), p2(0), p(0);
  real dPoint, dWeight;
  real dIntegral;

  // Set the first weight (won't be used)
  dWeightFunctionNodalValues[0][0][0] = 1.0;
  
  // Compute the weights for every order
  for (int i=1;i<(iHighestOrder+1);i++){

	 // Set the first weight (won't be used)
	 dWeightFunctionNodalValues[i][0][0] = 1.0;
	 
	 // Computing weights for order q = i

	 // Set the matrix dimensions
	 A = new Matrix<real>(i,i);
	 B = new Matrix<real>(i,i);
	 
	 // Compute the matrix coefficients
	 for (int j=0;j<i;j++)
		for (int k=0;k<i;k++){

		  // Compute the integral by quadrature
		  dIntegral = 0.0;

		  for (int l=0;l<=i;l++){
			 dPoint  = dNodalPoints[i][l];
			 dWeight = qQuadrature->GetWeight(i+1,l) * 0.5;
				
			 dIntegral += ( lBasisFunctions[i][k+1]->Derivative(dPoint) *
								 lBasisFunctions[i-1][j]->Value(dPoint) *
								 dWeight );
			 
		  }


		  A->Set(j,k,dIntegral);
		  
		}

	 // Compute the inverse
	 *B = A->InverseHighPrecision();
	 
	 // Compute the nodal weights
	 for (int j=1;j<(i+1);j++){

		for (int k=0;k<(i+1);k++){

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
	 for (int j=1;j<(i+1);j++)
		for (int k=0;k<i;k++)
		  dWeightFunctionNodalValues[i][j][k] = B->Get(j-1,k);
	 
	 // Set the weight for j=0 which is not used for cG(q)
	 for (int k=0;k<=i;k++)
		dNodalWeights[i][0][k] = 0.0;
	 
	 // Delete the matrix
	 delete A;
	 delete B;
	 
  } 

}
//-----------------------------------------------------------------------------
void cGqMethod::ComputeInterpolationConstants()
{
  // This function computes the interpolation constants for the
  // cG(q) method

  // Set the q = 0 constant
  dInterpolationConstants[0] = 1.0;

  // Set the rest of the constants
  for (int i=1;i<=(iHighestOrder+1);i++)
	 dInterpolationConstants[i] = 1.0 / ( pow(2.0,real(i-1)) * Factorial(i-1) );
}
//-----------------------------------------------------------------------------
void cGqMethod::ComputeResidualFactors()
{
  // This function computes the residual factors, i.e.
  //
  //             mean(|r|) = c * |r|(endtime)
  //
  // assuming that the residual is a legendre polynomial.
  
  real dIntegral;
  real x;
  real dx = DEFAULT_RESIDUAL_FACTOR_STEP;
  real f;
  Legendre p;
  
  // Set the first factor to 1, not used for cG(q)
  dResidualFactors[0] = 1.0;
  
  for (int i=1;i<=iHighestOrder;i++){

	 // Check for precalc
	 if ( i <= PRECALC_MAX ){
		dResidualFactors[i] = dPrecalcResidualFactors_cG[i];
		continue;
	 }
	 
	 dIntegral = 0.0;

	 for (x=(-1.0+dx/2.0);x<1.0;x+=dx){

		f = fabs(p.Value(i,x));

		dIntegral += f*dx;
		
	 }
	 
	 dIntegral /= 2.0;

	 dResidualFactors[i] = dIntegral / 1.0;

  }

}
//-----------------------------------------------------------------------------
void cGqMethod::ComputeProductFactors()
{
  // This function computes the residual factors, i.e.
  //
  //             mean(|r * (v-w)| = c * |(r*(v-w))(endtime)|
  //
  // where r is the residual, v is the dual and w is the interpolant,
  // assuming that r and (v-w) are legendre polynomials. This is
  // (approximately) the case if w is chosen as the interpolant at the
  // Gauss quadrature points.
  
  real dIntegral;
  real x;
  real dx = DEFAULT_RESIDUAL_FACTOR_STEP;
  real f;
  Legendre p;
  
  // Set the first factor to 1, not used for cG(q)
  dResidualFactors[0] = 1.0;
  
  for (int i=1;i<=iHighestOrder;i++){

	 // Check for precalc
	 if ( i <= PRECALC_MAX ){
		dProductFactors[i] = dPrecalcProductFactors_cG[i];
		continue;
	 }
	 
	 dIntegral = 0.0;

	 for (x=(-1.0+dx/2.0);x<1.0;x+=dx){

		f = fabs(p.Value(i,x));
		f = f * f;
		
		// Compute the square
		
		dIntegral += f*dx;
		
	 }
	 
	 dIntegral /= 2.0;

	 dProductFactors[i] = dIntegral / 1.0;
	 
  }
}
//-----------------------------------------------------------------------------
void cGqMethod::ComputeQuadratureFactors()
{
  // This function computes the quadrature factors for estimating
  // the quadrature error in terms of quadrature differences,
  //
  // |E_i| = c * |E_i - E_{i+1}|

  real n;
  
  // Set the first factor, won't be used
  dQuadratureFactors[0] = 1.0;

  // Set the rest
  for (int i=1;i<=iHighestOrder;i++){
	 n = real(i + 1);
	 dQuadratureFactors[i] = 1.0 / ( 1.0 - pow(2.0,3.0-2.0*n) );
  }

}
//-----------------------------------------------------------------------------
void cGqMethod::ComputeInterpolationWeights()
{
  // This function computes the values of the lagrange basis functions
  // of order q-1, located at the q Gauss points.

  Lagrange *p;
  real dThisPoint, dPoint;
  real c;
  int iPosition;
  
  // Set the point and weight for q = 0, won't be used
  dInterpolationWeights[0][0] = 1.0;
  dInterpolationPoints[0][0]  = 0.5;

  if ( iHighestOrder > 0 ){
	 // Set the point and weight for q = 1, constant interpolant
	 dInterpolationWeights[1][0] = 1.0;
	 dInterpolationPoints[1][0]  = 0.5;
  }
	 
  // Compute the polynomials for every order greater than 1
  for (int i=2;i<(iHighestOrder+1);i++){

	 // Make a new Lagrange polynomial
	 p = new Lagrange(i-1);
	 
	 for (int j=0;j<i;j++){

		// This nodal point
		dThisPoint = qGauss->GetPoint(i,j);
		
		// Compute polynomial j for order i
		
		// First compute the constant
		c = 1.0;
		for (int k=0;k<i;k++)
		  if ( k != j ){
			 dPoint = qGauss->GetPoint(i,k);
			 c *= ( dThisPoint - dPoint );
		  }
		c = 1/c;
		
		// Set coefficients for Lagrange basis function
		iPosition = 0;
		for (int k=0;k<i;k++)
		  if ( k != j )
			 p->SetPoint(iPosition++,qGauss->GetPoint(i,k));
		p->SetConstant(c);

		// Evaluate at the endpoint
		dInterpolationWeights[i][j] = p->Value(1.0);

		// Set the nodal point
		dInterpolationPoints[i][j] = (dThisPoint + 1.0) / 2.0;
		
	 }

	 // Delete the polynomial
	 delete p;
	 
  }
  
}
//-----------------------------------------------------------------------------
