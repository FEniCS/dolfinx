// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Contributions by: Georgios Foufas 2002, 2003

#include <dolfin/DirectSolver.h>
#include <dolfin/KrylovSolver.h>
#include <dolfin/DenseMatrix.h>
#include <dolfin/SparseMatrix.h>
#include <dolfin/Matrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Matrix::Matrix(Type type = SPARSE)
{
  if ( type == DENSE )
    A = new DenseMatrix();
  else
    A = new SparseMatrix();
  
  _type = type;
}
//-----------------------------------------------------------------------------
Matrix::Matrix(int m, int n, Type type = SPARSE)
{
  if ( type = DENSE )
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
Matrix::Element Matrix::operator()(int i, int j)
{
  return Element(*this, i, j);
}
//-----------------------------------------------------------------------------
real Matrix::operator()(int i, int j) const
{
  return (*A)(i,j);
}
//-----------------------------------------------------------------------------
real Matrix::operator()(int i, int& j, int pos) const
{
  return (*A)(i,j,pos);
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
real Matrix::mult(Vector& x, int i) const
{
  return A->mult(x,i);
}
//-----------------------------------------------------------------------------
void Matrix::mult(Vector& x, Vector& Ax) const
{
  A->mult(x,Ax);
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
int Matrix::perm(int i) const
{
  return A->perm(i);
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
Matrix::Row::Row(Matrix& matrix, int i) : A(matrix)
{
  this->i = i;
}
//-----------------------------------------------------------------------------   
Matrix::Column::Column(Matrix& matrix, int j) : A(matrix)
{
  this->j = j;
}
//-----------------------------------------------------------------------------
real** Matrix::values()
{
  // Matrix::values() and GenericMatrix::values() are the "same" functions
  // but have different names, since otherwise there would be a conflict
  // with the variable values in DenseMatrix and SparseMatrix.

  return A->getvalues();
}
//-----------------------------------------------------------------------------
int* Matrix::permutation()
{
  // Set comment in Matrix::values() above
  
  return A->getperm();
}
//-----------------------------------------------------------------------------
