// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Karin Kraft, 2004.
// Modified by Erik Svensson, 2004.

#include <dolfin/GenericMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericMatrix::GenericMatrix():
  m(0),
  n(0)
{}
//-----------------------------------------------------------------------------
GenericMatrix::GenericMatrix(unsigned int m, unsigned int n)
{
  this->m = m;
  this->n = n;
}
//-----------------------------------------------------------------------------
GenericMatrix::~GenericMatrix()
{}
//-----------------------------------------------------------------------------
void GenericMatrix::init(unsigned int m, unsigned int n)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::clear()
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
unsigned int GenericMatrix::size(unsigned int dim) const
{
  if ( dim == 0 )
    return m;
  else if ( dim == 1 )
    return n;
  
  dolfin_warning1("Illegal matrix dimension: dim = %d.", dim);
  return 0;
}
//-----------------------------------------------------------------------------
unsigned int GenericMatrix::size() const
{
  dolfin_error("This function is not implemented");
  return 0;
}
//-----------------------------------------------------------------------------
unsigned int GenericMatrix::rowsize(unsigned int i) const
{
  dolfin_error("This function is not implemented");
  return 0;
}
//-----------------------------------------------------------------------------
unsigned int GenericMatrix::bytes() const
{
  dolfin_error("This function is not implemented");
  return 0;
}
//-----------------------------------------------------------------------------
real GenericMatrix:: operator()(unsigned int i, unsigned int j) const    
{
  dolfin_error("This function is not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
real* GenericMatrix::operator[](unsigned int i)
{
  dolfin_error("This function is not implemented");
  return 0;
}
//-----------------------------------------------------------------------------
real GenericMatrix:: operator()(unsigned int i, unsigned int& j, unsigned int pos) const
{
  dolfin_error("This function is not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
real& GenericMatrix:: operator()(unsigned int i, unsigned int& j, unsigned int pos)
{
  dolfin_error("This function is not implemented");
  return *(new real);
}
//-----------------------------------------------------------------------------
void GenericMatrix::operator= (real a)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::operator= (const DenseMatrix& A)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::operator= (const SparseMatrix& A)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::operator= (const GenericMatrix& A)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::operator+= (const DenseMatrix& A)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::operator+= (const SparseMatrix& A)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::operator+= (const GenericMatrix& A)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::operator-= (const DenseMatrix& A)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::operator-= (const SparseMatrix& A)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::operator-= (const GenericMatrix& A)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::operator*= (real a)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
real GenericMatrix::norm() const
{
  dolfin_error("This function is not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
real GenericMatrix::mult(const Vector& x, unsigned int i) const
{
  dolfin_error("This function is not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
void GenericMatrix::mult(const Vector& x, Vector& Ax) const
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::multt(const Vector& x, Vector& Ax) const
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::mult(const DenseMatrix& B, DenseMatrix& AB) const
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::mult(const SparseMatrix& B, SparseMatrix& AB) const
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::mult(const GenericMatrix& B, GenericMatrix& AB) const
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
real GenericMatrix::multrow(const Vector& x, unsigned int i) const
{
  dolfin_error("This function is not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
real GenericMatrix::multcol(const Vector& x, unsigned int j) const
{
  dolfin_error("This function is not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
void GenericMatrix::resize()
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::ident(unsigned int i)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::lump(Vector& a) const
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::addrow()
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::addrow(const Vector& x)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::initrow(unsigned int i, unsigned int rowsize)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
bool GenericMatrix::endrow(unsigned int i, unsigned int pos) const
{
  dolfin_error("This function is not implemented");
  return true;
}
//-----------------------------------------------------------------------------
void GenericMatrix::settransp(const DenseMatrix& A)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::settransp(const SparseMatrix& A)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::settransp(const GenericMatrix& A)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
real GenericMatrix::rowmax(unsigned int i) const
{
  dolfin_error("This function is not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
real GenericMatrix::colmax(unsigned int i) const
{
  dolfin_error("This function is not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
real GenericMatrix::rowmin(unsigned int i) const
{
  dolfin_error("This function is not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
real GenericMatrix::colmin(unsigned int i) const
{
  dolfin_error("This function is not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
real GenericMatrix::rowsum(unsigned int i) const
{
  dolfin_error("This function is not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
real GenericMatrix::colsum(unsigned int i) const
{
  dolfin_error("This function is not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
real GenericMatrix::rownorm(unsigned int i, unsigned int type) const
{
  dolfin_error("This function is not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
real GenericMatrix::colnorm(unsigned int i, unsigned int type) const
{
  dolfin_error("This function is not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
void GenericMatrix::show() const
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<< (LogStream& stream, const GenericMatrix& A)
{
  stream << "[ Generic matrix of size " 
	 << A.size(0) << " x " << A.size(1) << " ]";
  
  return stream;
}
//-----------------------------------------------------------------------------
void GenericMatrix::alloc(unsigned int m, unsigned int n)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
real GenericMatrix::read(unsigned int i, unsigned int j) const
{
  dolfin_error("This function is not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
void GenericMatrix::write(unsigned int i, unsigned int j, real value)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::add(unsigned int i, unsigned int j, real value)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::sub(unsigned int i, unsigned int j, real value)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::mult(unsigned int i, unsigned int j, real value)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::div(unsigned int i, unsigned int j, real value)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
real** GenericMatrix::getvalues()
{
  dolfin_error("This function is not implemented");
  return 0;
}
//-----------------------------------------------------------------------------
real** const GenericMatrix::getvalues() const
{
  dolfin_error("This function is not implemented");
  return 0;
}
//-----------------------------------------------------------------------------
void GenericMatrix::initperm()
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::clearperm()
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
unsigned int* GenericMatrix::getperm()
{
  dolfin_error("This function is not implemented");
  return 0;
}
//-----------------------------------------------------------------------------
unsigned int* const GenericMatrix::getperm() const
{
  dolfin_error("This function is not implemented");
  return 0;
}
//-----------------------------------------------------------------------------
