#include "lagrange.hh"

//----------------------------------------------------------------------------
Lagrange::Lagrange(int iiDegree)
{
  // Check that iiDegree >= 0
  assert( iiDegree >= 0 );
  
  // Set the degree
  iDegree = iiDegree;
  
  // Allocate memory
  if ( iiDegree > 0 ){
	 dPoints = new double[iiDegree];
  
	 // Check the allocated memory
	 assert( dPoints != 0 );
 
	 // Set the coefficients to zero
	 for (int i=0;i<iDegree;i++)
		dPoints[i] = 0.0;

	 dConstant = 0.0;
	 
  }
	 
}
//----------------------------------------------------------------------------
Lagrange::Lagrange(const Lagrange &p)
{
  // This is the copy constructor

  if ( p.iDegree > 0 ){
	 iDegree = p.iDegree;
	 if ( p.dPoints && p.iDegree >= 0 ){
		dPoints = new double[p.iDegree];
		for (int i=0;i<=p.iDegree;i++)
		  dPoints[i] = p.dPoints[i];
	 }
	 else
		dPoints = NULL;
  }

  dConstant = p.dConstant;
  
}
//----------------------------------------------------------------------------
Lagrange::~Lagrange()
{
  // Delete the coefficients
  if ( iDegree > 0 )
	 delete dPoints;
}
//----------------------------------------------------------------------------
void Lagrange::SetPoint(int iIndex, double dPoint)
{
  // Check the index
  assert( iIndex >= 0 );
  assert( iIndex < iDegree );

  // Set the new value
  dPoints[iIndex] = dPoint;
}
//----------------------------------------------------------------------------
void Lagrange::SetConstant(double ddConstant)
{
  // This function sets the constant

  dConstant = ddConstant;
}
//----------------------------------------------------------------------------
int Lagrange::GetDegree()
{
  // This function returns the degree.

  return ( iDegree );
}
//----------------------------------------------------------------------------
double Lagrange::GetPoint(int iIndex)
{
  // This function returns point iIndex.

  // Check the index
  assert( iIndex >= 0 );
  assert( iIndex < iDegree );
  
  return ( dPoints[iIndex] );
}
//----------------------------------------------------------------------------
double Lagrange::GetConstant()
{
  // This function returns the constant.

  return ( dConstant );
}
//----------------------------------------------------------------------------
double Lagrange::Value(double dPoint)
{
  // This function returns the value at dPoint.

  if ( iDegree == 0 )
	 return ( dConstant );
  
  double dValue = dConstant;

  for (int i=0;i<iDegree;i++)
	 dValue *= (dPoint - dPoints[i]);

  return ( dValue );
}
//----------------------------------------------------------------------------
double Lagrange::Derivative(double dPoint)
{
  // This function returns the value of the derivative at dPoint;

  if ( iDegree == 0 )
	 return 0.0;
  
  double dSum = 0.0;
  double dValue;
  
  for (int i=0;i<iDegree;i++){
	 dValue = 1.0;

	 for (int j=0;j<iDegree;j++)
		if (j!=i)
		  dValue *= (dPoint - dPoints[j]);

	 dSum += dValue;
  }

  return ( dConstant * dSum );
}
//----------------------------------------------------------------------------
double Lagrange::NthDerivative()
{
  // This function returns the iDegree:th derivative, which is a constant.

  double dValue = dConstant;
  
  for (int i=1;i<=iDegree;i++)
	 dValue *= double(i);
  
  return ( dValue );
}
//----------------------------------------------------------------------------
