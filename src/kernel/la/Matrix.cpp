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
  if ( type == DENSE )
    A = new DenseMatrix();
  else
    A = new SparseMatrix();
  
  _type = type;
}
//-----------------------------------------------------------------------------
Matrix::Matrix(int m, int n, Type type)
{
  if ( type == DENSE )
    A = new DenseMatrix(m,n);
  else
    A = new SparseMatrix(m,n);

  _type = type;
}
//-----------------------------------------------------------------------------
Matrix::Matrix(const Matrix& A)
{
  if ( A._type == DENSE )
    this->A = new DenseMatrix(*((DenseMatrix *) A.A));
  else
    this->A = new SparseMatrix(*((SparseMatrix *) A.A));

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
void Matrix::init(int m, int n)
{
  A->init(m,n);
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
int Matrix::size(int dim) const
{
  return A->size(dim);
}
//-----------------------------------------------------------------------------
int Matrix::size() const
{
  return A->size();
}
//-----------------------------------------------------------------------------
int Matrix::rowsize(int dim) const
{
  return A->rowsize(dim);
}
//-----------------------------------------------------------------------------
int Matrix::bytes() const
{
  return A->bytes();
}
//-----------------------------------------------------------------------------
real Matrix::operator()(int i, int j) const
{
  // This operator is used when the object is const
  return (*A)(i,j);
}
//-----------------------------------------------------------------------------
Matrix::Element Matrix::operator()(int i, int j)
{
  // This operator is used when the object is non-const and is slower
  return Element(*this, i, j);
}
//-----------------------------------------------------------------------------
Matrix::Row Matrix::operator()(int i, MatrixRange j)
{
  return Row(*this, i, j);
}
//-----------------------------------------------------------------------------
Matrix::Row Matrix::operator()(MatrixIndex i, MatrixRange j)
{
  return Row(*this, i, j);
}
//-----------------------------------------------------------------------------
Matrix::Column Matrix::operator()(MatrixRange i, int j)
{
  return Column(*this, i, j);
}
//-----------------------------------------------------------------------------
Matrix::Column Matrix::operator()(MatrixRange i, MatrixIndex j)
{
  return Column(*this, i, j);
}
//-----------------------------------------------------------------------------
real Matrix::operator()(int i, int& j, int pos) const
{
  return (*A)(i,j,pos);
}
//-----------------------------------------------------------------------------
real* Matrix::operator[](int i) const
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
  if ( A._type == DENSE )
    *(this->A) = *((DenseMatrix *) A.A);
  else
    *(this->A) = *((SparseMatrix *) A.A);
}
//-----------------------------------------------------------------------------
void Matrix::operator+=(const Matrix& A)
{
  if ( A._type == DENSE )
    *(this->A) += *((DenseMatrix *) A.A);
  else
    *(this->A) += *((SparseMatrix *) A.A);
}
//-----------------------------------------------------------------------------
void Matrix::operator-=(const Matrix& A)
{
  if ( A._type == DENSE )
    *(this->A) -= *((DenseMatrix *) A.A);
  else
    *(this->A) -= *((SparseMatrix *) A.A);
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
real Matrix::mult(const Vector& x, int i) const
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
real Matrix::multrow(const Vector& x, int i) const
{
  return A->multrow(x,i);
}
//-----------------------------------------------------------------------------
real Matrix::multcol(const Vector& x, int j) const
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
  
  if ( _type == DENSE ) {
    DirectSolver solver;
    solver.solve(*this, x, b);
  }
  else {
    KrylovSolver solver;
    solver.solve(*this, x, b);
  }
}
//-----------------------------------------------------------------------------
void Matrix::inverse(Matrix& Ainv)
{
  if ( _type == SPARSE )
    dolfin_error("Not implemented for sparse matrices. Consider using dense().");

  DirectSolver solver;
  solver.inverse(*this, Ainv);
}
//-----------------------------------------------------------------------------
void Matrix::hpsolve(Vector& x, const Vector& b) const
{
  if ( _type == SPARSE )
    dolfin_error("Not implemented for sparse matrices. Consider using dense().");

  DirectSolver solver;
  solver.hpsolve(*this, x, b);
}
//-----------------------------------------------------------------------------
void Matrix::lu()
{
  if ( _type == SPARSE )
    dolfin_error("Not implemented for sparse matrices. Consider using dense().");
  
  DirectSolver solver;
  solver.lu(*this);
}
//-----------------------------------------------------------------------------
void Matrix::solveLU(Vector& x, const Vector& b) const
{
  if ( _type == SPARSE )
    dolfin_error("Not implemented for sparse matrices. Consider using dense().");

  DirectSolver solver;
  solver.solveLU(*this, x, b);
}
//-----------------------------------------------------------------------------
void Matrix::inverseLU(Matrix& Ainv) const
{
  if ( _type == SPARSE )
    dolfin_error("Not implemented for sparse matrices. Consider using dense().");

  DirectSolver solver;
  solver.inverseLU(*this, Ainv);
}
//-----------------------------------------------------------------------------
void Matrix::hpsolveLU(const Matrix& LU, Vector& x, const Vector& b) const
{
  if ( _type == SPARSE )
    dolfin_error("Not implemented for sparse matrices. Consider using dense().");

  DirectSolver solver;
  solver.hpsolveLU(LU, *this, x, b);
}
//-----------------------------------------------------------------------------
void Matrix::resize()
{
  A->resize();
}
//-----------------------------------------------------------------------------
void Matrix::ident(int i)
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
void Matrix::initrow(int i, int rowsize)
{
  A->initrow(i, rowsize);
}
//-----------------------------------------------------------------------------
bool Matrix::endrow(int i, int pos) const
{
  return A->endrow(i, pos);
}
//-----------------------------------------------------------------------------
void Matrix::settransp(const Matrix& A)
{
  if ( A._type == DENSE )
    this->A->settransp(*((DenseMatrix *) A.A));
  else
    this->A->settransp(*((SparseMatrix *) A.A));
}
//-----------------------------------------------------------------------------
void Matrix::show() const
{
  A->show();
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const Matrix& A)
{
  if ( A.type() == Matrix::DENSE )
    stream << *((DenseMatrix *) A.A);
  else
    stream << *((SparseMatrix *) A.A);

  return stream;
}
//-----------------------------------------------------------------------------
Matrix::Element::Element(Matrix& matrix, int i, int j) : A(matrix)
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
Matrix::Row::Row(Matrix& matrix, int i, MatrixRange) : A(matrix)
{
  this->i = i;
  this->j = j;
}
//-----------------------------------------------------------------------------
Matrix::Row::Row(Matrix& matrix, MatrixIndex i, MatrixRange j) : A(matrix)
{
  if ( i == first )
    this->i = 0;
  else
    this->i = A.size(0) - 1;

  this->j = j;
}
//-----------------------------------------------------------------------------
int Matrix::Row::size() const
{
  return A.size(1);
}
//-----------------------------------------------------------------------------
real Matrix::Row::operator()(int j) const
{
  return A(i,j);
}
//-----------------------------------------------------------------------------
Matrix::Element Matrix::Row::operator()(int j)
{
  return Element(A, i, j);
}
//-----------------------------------------------------------------------------
void Matrix::Row::operator=(const Row& row)
{
  if ( A.size(1) != row.A.size(1) )
    dolfin_error("Matrix dimensions don't match.");
  
  for (int j = 0; j < A.size(1); j++)
    A(i,j) = row.A(row.i,j);
}
//-----------------------------------------------------------------------------
void Matrix::Row::operator=(const Column& col)
{
  if ( A.size(1) != col.A.size(0) )
    dolfin_error("Matrix dimensions don't match.");

  for (int j = 0; j < A.size(1); j++)
    A(i,j) = col.A(j,col.j);
}
//-----------------------------------------------------------------------------
void Matrix::Row::operator=(const Vector& x)
{
  if ( x.size() != A.size(1) )
    dolfin_error("Matrix imensions don't match.");

  for (int j = 0; j < x.size(); j++)
    A(i,j) = x(j);
}
//-----------------------------------------------------------------------------
real Matrix::Row::operator* (const Vector& x) const
{
  return A.multrow(x,i);
}
//-----------------------------------------------------------------------------
Matrix::Column::Column(Matrix& matrix, MatrixRange i, int j) : A(matrix)
{
  this->i = i;
  this->j = j;
}
//-----------------------------------------------------------------------------
Matrix::Column::Column(Matrix& matrix, MatrixRange i, MatrixIndex j) : A(matrix)
{
  this->i = i;
  
  if ( j == first )
    this->j = 0;
  else
    this->j = A.size(1) - 1;
}
//-----------------------------------------------------------------------------
int Matrix::Column::size() const
{
  return A.size(0);
}
//-----------------------------------------------------------------------------
real Matrix::Column::operator()(int i) const
{
  return A(i,j);
}
//-----------------------------------------------------------------------------
Matrix::Element Matrix::Column::operator()(int i)
{
  return Element(A, i, j);
}
//-----------------------------------------------------------------------------
void Matrix::Column::operator=(const Column& col)
{
  if ( A.size(0) != col.A.size(0) )
    dolfin_error("Matrix dimensions don't match.");

  for (int i = 0; i < A.size(0); i++)
    A(i,j) = col.A(i,j);
}
//-----------------------------------------------------------------------------
void Matrix::Column::operator=(const Row& row)
{
  if ( A.size(0) != row.A.size(1) )
    dolfin_error("Matrix dimensions don't match.");

  for (int i = 0; i < A.size(0); i++)
    A(i,j) = row.A(row.i,i);
}
//-----------------------------------------------------------------------------
void Matrix::Column::operator=(const Vector& x)
{
  if ( x.size() != A.size(0) )
    dolfin_error("Matrix imensions don't match.");

  for (int i = 0; i < x.size(); i++)
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
int* Matrix::permutation()
{
  // See comment in Matrix::values() above
  
  return A->getperm();
}
//-----------------------------------------------------------------------------
int* const Matrix::permutation() const
{
  // See comment in Matrix::values() above
  
  return A->getperm();
}
//-----------------------------------------------------------------------------
