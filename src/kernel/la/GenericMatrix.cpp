// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/GenericMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericMatrix::GenericMatrix()
{
  m = 0;
  n = 0;
}
//-----------------------------------------------------------------------------
GenericMatrix::GenericMatrix(int m, int n)
{
  this->m = m;
  this->n = n;
}
//-----------------------------------------------------------------------------
GenericMatrix::~GenericMatrix()
{

}
//-----------------------------------------------------------------------------
void GenericMatrix::init(int m, int n)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::clear()
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
int GenericMatrix::size(int dim) const
{
  if ( dim == 0 )
    return m;
  else if ( dim == 1 )
    return n;
  
  dolfin_warning1("Illegal matrix dimension: dim = %d.", dim);
  return 0;
}
//-----------------------------------------------------------------------------
int GenericMatrix::size() const
{
  dolfin_error("This function is not implemented");
  return 0;
}
//-----------------------------------------------------------------------------
int GenericMatrix::rowsize(int i) const
{
  dolfin_error("This function is not implemented");
  return 0;
}
//-----------------------------------------------------------------------------
int GenericMatrix::bytes() const
{
  dolfin_error("This function is not implemented");
  return 0;
}
//-----------------------------------------------------------------------------
real GenericMatrix:: operator()(int i, int j) const    
{
  dolfin_error("This function is not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
real* GenericMatrix::operator[](int i)
{
  dolfin_error("This function is not implemented");
  return 0;
}
//-----------------------------------------------------------------------------
real GenericMatrix:: operator()(int i, int& j, int pos) const
{
  dolfin_error("This function is not implemented");
  return 0.0;
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
real GenericMatrix::mult(const Vector& x, int i) const
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
real GenericMatrix::multrow(const Vector& x, int i) const
{
  dolfin_error("This function is not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
real GenericMatrix::multcol(const Vector& x, int j) const
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
void GenericMatrix::ident(int i)
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
void GenericMatrix::initrow(int i, int rowsize)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
bool GenericMatrix::endrow(int i, int pos) const
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
void GenericMatrix::alloc(int m, int n)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
real GenericMatrix::read(int i, int j) const
{
  dolfin_error("This function is not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
void GenericMatrix::write(int i, int j, real value)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::add(int i, int j, real value)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::sub(int i, int j, real value)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::mult(int i, int j, real value)
{
  dolfin_error("This function is not implemented");
}
//-----------------------------------------------------------------------------
void GenericMatrix::div(int i, int j, real value)
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
int* GenericMatrix::getperm()
{
  dolfin_error("This function is not implemented");
  return 0;
}
//-----------------------------------------------------------------------------
int* const GenericMatrix::getperm() const
{
  dolfin_error("This function is not implemented");
  return 0;
}
//-----------------------------------------------------------------------------
