// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Contributions by: Georgios Foufas 2002, 2003

#include <dolfin/DirectSolver.h>
#include <dolfin/KrylovSolver.h>
#include <dolfin/Vector.h>
#include <dolfin/DenseMatrix.h>
#include <dolfin/SparseMatrix.h>
#include <dolfin/Matrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Matrix::Matrix(Type type)
{
  switch ( type ) {
  case dense:
    A = new DenseMatrix();
    break;
  case sparse:
    A = new SparseMatrix();
    break;
  default:
    A = new GenericMatrix();
  }
  
  _type = type;
}
//-----------------------------------------------------------------------------
Matrix::Matrix(unsigned int m, unsigned int n, Type type)
{
  switch ( type ) {
  case dense:
    A = new DenseMatrix(m,n);
    break;
  case sparse:
    A = new SparseMatrix(m,n);
    break;
  default:
    A = new GenericMatrix(m,n);
  }

  _type = type;
}
//-----------------------------------------------------------------------------
Matrix::Matrix(const Matrix& A)
{
  switch ( A._type ) {
  case dense:
    this->A = new DenseMatrix(*((DenseMatrix *) A.A));
    break;
  case sparse:
    this->A = new SparseMatrix(*((SparseMatrix *) A.A));
    break;
  default:
    this->A = new GenericMatrix(A.size(0), A.size(1));
  }
  
  _type = A._type;
}
//-----------------------------------------------------------------------------
Matrix::~Matrix ()
{
  if ( A )
    delete A;
  A = 0;
}
//-----------------------------------------------------------------------------
void Matrix::init(unsigned int m, unsigned int n, Type type)
{
  dolfin_assert(A);

  // Check if we need to change the type
  if ( _type != type )
  {
    switch ( type ) {
    case dense:
      A = new DenseMatrix(m, n);
      break;
    case sparse:
      A = new SparseMatrix(m, n);
      break;
    default:
      A = new GenericMatrix(m, n);
    }
  }
  else
    A->init(m, n);
}
//-----------------------------------------------------------------------------
void Matrix::clear()
{
  A->clear();
}
//-----------------------------------------------------------------------------
Matrix::Type Matrix::type() const
{
  return _type;
}
//-----------------------------------------------------------------------------
unsigned int Matrix::size(unsigned int dim) const
{
  return A->size(dim);
}
//-----------------------------------------------------------------------------
unsigned int Matrix::size() const
{
  return A->size();
}
//-----------------------------------------------------------------------------
unsigned int Matrix::rowsize(unsigned int dim) const
{
  return A->rowsize(dim);
}
//-----------------------------------------------------------------------------
unsigned int Matrix::bytes() const
{
  return A->bytes();
}
//-----------------------------------------------------------------------------
real Matrix::operator()(unsigned int i, unsigned int j) const
{
  // This operator is used when the object is const
  return (*A)(i,j);
}
//-----------------------------------------------------------------------------
Matrix::Element Matrix::operator()(unsigned int i, unsigned int j)
{
  // This operator is used when the object is non-const and is slower
  return Element(*this, i, j);
}
//-----------------------------------------------------------------------------
Matrix::Row Matrix::operator()(unsigned int i, Range j)
{
  return Row(*this, i, j);
}
//-----------------------------------------------------------------------------
Matrix::Row Matrix::operator()(Index i, Range j)
{
  return Row(*this, i, j);
}
//-----------------------------------------------------------------------------
Matrix::Column Matrix::operator()(Range i, unsigned int j)
{
  return Column(*this, i, j);
}
//-----------------------------------------------------------------------------
Matrix::Column Matrix::operator()(Range i, Index j)
{
  return Column(*this, i, j);
}
//-----------------------------------------------------------------------------
real Matrix::operator()(unsigned int i, unsigned int& j, unsigned int pos) const
{
  return (*A)(i,j,pos);
}
//-----------------------------------------------------------------------------
real* Matrix::operator[](unsigned int i) const
{
  return (*A)[i];
}
//-----------------------------------------------------------------------------
void Matrix::operator=(real a)
{
  (*A) = 0.0;
}
//-----------------------------------------------------------------------------
void Matrix::operator=(const Matrix& A)
{
  switch ( A._type == dense ) {
  case dense:
    *(this->A) = *((DenseMatrix *) A.A);
    break;
  case sparse:
    *(this->A) = *((SparseMatrix *) A.A);
    break;
  default:
    *(this->A) = *((GenericMatrix *) A.A);
  }
}
//-----------------------------------------------------------------------------
void Matrix::operator+=(const Matrix& A)
{
  switch ( A._type ) {
  case dense:
    *(this->A) += *((DenseMatrix *) A.A);
    break;
  case sparse:
    *(this->A) += *((SparseMatrix *) A.A);
    break;
  default:
    *(this->A) += *((GenericMatrix *) A.A);
  }
}
//-----------------------------------------------------------------------------
void Matrix::operator-=(const Matrix& A)
{
  switch ( A._type ) {
  case dense:
    *(this->A) -= *((DenseMatrix *) A.A);
    break;
  case sparse:
    *(this->A) -= *((SparseMatrix *) A.A);
    break;
  default:
    *(this->A) -= *((GenericMatrix *) A.A);
  }
}
//-----------------------------------------------------------------------------
void Matrix::operator*=(real a)
{
  *(this->A) *= a;
}
//-----------------------------------------------------------------------------
real Matrix::norm() const
{
  return A->norm();
}
//-----------------------------------------------------------------------------
real Matrix::mult(const Vector& x, unsigned int i) const
{
  return A->mult(x,i);
}
//-----------------------------------------------------------------------------
void Matrix::mult(const Vector& x, Vector& Ax) const
{
  A->mult(x,Ax);
}
//-----------------------------------------------------------------------------
void Matrix::multt(const Vector& x, Vector &Ax) const
{
  A->multt(x,Ax);
}
//-----------------------------------------------------------------------------
real Matrix::multrow(const Vector& x, unsigned int i) const
{
  return A->multrow(x,i);
}
//-----------------------------------------------------------------------------
real Matrix::multcol(const Vector& x, unsigned int j) const
{
  return A->multcol(x,j);
}
//-----------------------------------------------------------------------------
void Matrix::transp(Matrix& A) const
{
  A.settransp(*this);
}
//-----------------------------------------------------------------------------
void Matrix::solve(Vector& x, const Vector& b)
{
  // Note that these need to be handled here and not redirected to
  // GenericMatrix, since the solvers expect a Matrix as argument.
  // No one else but Matrix should be concerned with different types
  // of matrices.
  
  switch ( _type ) {
  case dense:
    {
      DirectSolver solver;
      solver.solve(*this, x, b);
    }
    break;
  case sparse:
    {
      KrylovSolver solver;
      solver.solve(*this, x, b);
    }
    break;
  default:
    {
      KrylovSolver solver;
      solver.solve(*this, x, b);
    }
  }
}
//-----------------------------------------------------------------------------
void Matrix::inverse(Matrix& Ainv)
{
  switch ( _type ) {
  case dense: 
    {
      DirectSolver solver;
      solver.inverse(*this, Ainv);
    }
    break;
  case sparse:
    dolfin_error("Not implemented for a sparse matrix. Consider using a dense matrix.");
    break;
  default:
    dolfin_error("Not implemented for a generic matrix. Consider using a dense matrix.");
  }
}
//-----------------------------------------------------------------------------
void Matrix::hpsolve(Vector& x, const Vector& b) const
{
  switch ( _type ) {
  case dense:
    {
      DirectSolver solver;
      solver.hpsolve(*this, x, b);
    }
    break;
  case sparse:
    dolfin_error("Not implemented for a sparse matrix. Consider using a dense matrix.");
    break;
  default:
    dolfin_error("Not implemented for a generic matrix. Consider using a dense matrix.");
  }
}
//-----------------------------------------------------------------------------
void Matrix::lu()
{
  switch ( _type ) {
  case dense:
    {
      DirectSolver solver;
      solver.lu(*this);
    }
    break;
  case sparse:
    dolfin_error("Not implemented for a sparse matrix. Consider using a dense matrix.");
    break;
  default:
    dolfin_error("Not implemented for a generic matrix. Consider using a dense matrix.");
  }
}
//-----------------------------------------------------------------------------
void Matrix::solveLU(Vector& x, const Vector& b) const
{
  switch ( _type ) {
  case dense:
    {
      DirectSolver solver;
      solver.solveLU(*this, x, b);
    }
    break;
  case sparse:
    dolfin_error("Not implemented for a sparse matrix. Consider using a dense matrix.");
    break;
  default:
    dolfin_error("Not implemented for a generic matrix. Consider using a dense matrix.");
  }
}
//-----------------------------------------------------------------------------
void Matrix::inverseLU(Matrix& Ainv) const
{
  switch ( _type ) {
  case dense:
    {
      DirectSolver solver;
      solver.inverseLU(*this, Ainv);
    }
    break;
  case sparse:
    dolfin_error("Not implemented for a sparse matrix. Consider using a dense matrix.");
    break;
  default:
    dolfin_error("Not implemented for a generic matrix. Consider using a dense matrix.");
  }
}
//-----------------------------------------------------------------------------
void Matrix::hpsolveLU(const Matrix& LU, Vector& x, const Vector& b) const
{
  switch ( _type ) {
  case dense:
    {
      DirectSolver solver;
      solver.hpsolveLU(LU, *this, x, b);
    }
    break;
  case sparse:
    dolfin_error("Not implemented for a sparse matrix. Consider using a dense matrix.");
    break;
  default:
    dolfin_error("Not implemented for a generic matrix. Consider using a dense matrix.");
  }
}
//-----------------------------------------------------------------------------
void Matrix::resize()
{
  A->resize();
}
//-----------------------------------------------------------------------------
void Matrix::ident(unsigned int i)
{
  A->ident(i);
}
//-----------------------------------------------------------------------------
void Matrix::addrow()
{
  A->addrow();
}
//-----------------------------------------------------------------------------
void Matrix::addrow(const Vector& x)
{
  A->addrow(x);
}
//-----------------------------------------------------------------------------
void Matrix::initrow(unsigned int i, unsigned int rowsize)
{
  A->initrow(i, rowsize);
}
//-----------------------------------------------------------------------------
bool Matrix::endrow(unsigned int i, unsigned int pos) const
{
  return A->endrow(i, pos);
}
//-----------------------------------------------------------------------------
void Matrix::settransp(const Matrix& A)
{
  switch ( A._type ) {
  case dense:
    this->A->settransp(*((DenseMatrix *) A.A));
    break;
  case sparse:
    this->A->settransp(*((SparseMatrix *) A.A));
    break;
  default:
    this->A->settransp(*((GenericMatrix *) A.A));
  }
}
//-----------------------------------------------------------------------------
void Matrix::show() const
{
  A->show();
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const Matrix& A)
{
  switch ( A.type() ) {
  case Matrix::dense:
    stream << *((DenseMatrix *) A.A);
    break;
  case Matrix::sparse:
    stream << *((SparseMatrix *) A.A);
    break;
  default:
    stream << *((GenericMatrix *) A.A);
  }

  return stream;
}
//-----------------------------------------------------------------------------
Matrix::Element::Element(Matrix& matrix, unsigned int i, unsigned int j) : A(matrix)
{
  this->i = i;
  this->j = j;
}
//-----------------------------------------------------------------------------
Matrix::Element::operator real() const
{
  //dolfin_debug("Conversion to real");

  return A.A->read(i,j);
}
//-----------------------------------------------------------------------------
void Matrix::Element::operator=(real a)
{
  A.A->write(i, j, a);
}
//-----------------------------------------------------------------------------
void Matrix::Element::operator=(const Element& e)
{
  A.A->write(i, j, e.A.A->read(e.i, e.j));
}
//-----------------------------------------------------------------------------
void Matrix::Element::operator+=(real a)
{
  A.A->add(i, j, a);
}
//-----------------------------------------------------------------------------
void Matrix::Element::operator-=(real a)
{
  A.A->sub(i, j, a);
}
//-----------------------------------------------------------------------------
void Matrix::Element::operator*=(real a)
{
  A.A->mult(i, j, a);
}
//-----------------------------------------------------------------------------
void Matrix::Element::operator/=(real a)
{
  A.A->div(i, j, a);
}
//-----------------------------------------------------------------------------
Matrix::Row::Row(Matrix& matrix, unsigned int i, Range) : A(matrix)
{
  this->i = i;
  this->j = j;
}
//-----------------------------------------------------------------------------
Matrix::Row::Row(Matrix& matrix, Index i, Range j) : A(matrix)
{
  if ( i == first )
    this->i = 0;
  else
    this->i = A.size(0) - 1;

  this->j = j;
}
//-----------------------------------------------------------------------------
unsigned int Matrix::Row::size() const
{
  return A.size(1);
}
//-----------------------------------------------------------------------------
real Matrix::Row::operator()(unsigned int j) const
{
  return A(i,j);
}
//-----------------------------------------------------------------------------
Matrix::Element Matrix::Row::operator()(unsigned int j)
{
  return Element(A, i, j);
}
//-----------------------------------------------------------------------------
void Matrix::Row::operator=(const Row& row)
{
  if ( A.size(1) != row.A.size(1) )
    dolfin_error("Matrix dimensions don't match.");
  
  for (unsigned int j = 0; j < A.size(1); j++)
    A(i,j) = row.A(row.i,j);
}
//-----------------------------------------------------------------------------
void Matrix::Row::operator=(const Column& col)
{
  if ( A.size(1) != col.A.size(0) )
    dolfin_error("Matrix dimensions don't match.");

  for (unsigned int j = 0; j < A.size(1); j++)
    A(i,j) = col.A(j,col.j);
}
//-----------------------------------------------------------------------------
void Matrix::Row::operator=(const Vector& x)
{
  if ( x.size() != A.size(1) )
    dolfin_error("Matrix imensions don't match.");

  for (unsigned int j = 0; j < x.size(); j++)
    A(i,j) = x(j);
}
//-----------------------------------------------------------------------------
real Matrix::Row::operator* (const Vector& x) const
{
  return A.multrow(x,i);
}
//-----------------------------------------------------------------------------
Matrix::Column::Column(Matrix& matrix, Range i, unsigned int j) : A(matrix)
{
  this->i = i;
  this->j = j;
}
//-----------------------------------------------------------------------------
Matrix::Column::Column(Matrix& matrix, Range i, Index j) : A(matrix)
{
  this->i = i;
  
  if ( j == first )
    this->j = 0;
  else
    this->j = A.size(1) - 1;
}
//-----------------------------------------------------------------------------
unsigned int Matrix::Column::size() const
{
  return A.size(0);
}
//-----------------------------------------------------------------------------
real Matrix::Column::operator()(unsigned int i) const
{
  return A(i,j);
}
//-----------------------------------------------------------------------------
Matrix::Element Matrix::Column::operator()(unsigned int i)
{
  return Element(A, i, j);
}
//-----------------------------------------------------------------------------
void Matrix::Column::operator=(const Column& col)
{
  if ( A.size(0) != col.A.size(0) )
    dolfin_error("Matrix dimensions don't match.");

  for (unsigned int i = 0; i < A.size(0); i++)
    A(i,j) = col.A(i,j);
}
//-----------------------------------------------------------------------------
void Matrix::Column::operator=(const Row& row)
{
  if ( A.size(0) != row.A.size(1) )
    dolfin_error("Matrix dimensions don't match.");

  for (unsigned int i = 0; i < A.size(0); i++)
    A(i,j) = row.A(row.i,i);
}
//-----------------------------------------------------------------------------
void Matrix::Column::operator=(const Vector& x)
{
  if ( x.size() != A.size(0) )
    dolfin_error("Matrix imensions don't match.");

  for (unsigned int i = 0; i < x.size(); i++)
    A(i,j) = x(i);
}
//-----------------------------------------------------------------------------
real Matrix::Column::operator* (const Vector& x) const
{
  return A.multcol(x,j);
}
//-----------------------------------------------------------------------------
real** Matrix::values()
{
  // Matrix::values() and GenericMatrix::getvalues() are the "same" functions
  // but have different names, since otherwise there would be a conflict
  // with the variable values in DenseMatrix and SparseMatrix.

  return A->getvalues();
}
//-----------------------------------------------------------------------------
real** const Matrix::values() const
{
  // Matrix::values() and GenericMatrix::getvalues() are the "same" functions
  // but have different names, since otherwise there would be a conflict
  // with the variable values in DenseMatrix and SparseMatrix.
  
  return A->getvalues();
}
//-----------------------------------------------------------------------------
void Matrix::initperm()
{
  A->initperm();
}
//-----------------------------------------------------------------------------
void Matrix::clearperm()
{
  A->clearperm();
}
//-----------------------------------------------------------------------------
unsigned int* Matrix::permutation()
{
  // See comment in Matrix::values() above
  
  return A->getperm();
}
//-----------------------------------------------------------------------------
unsigned int* const Matrix::permutation() const
{
  // See comment in Matrix::values() above
  
  return A->getperm();
}
//-----------------------------------------------------------------------------
