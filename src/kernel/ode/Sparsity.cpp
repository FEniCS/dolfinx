// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/AutomaticSparsity.h>
#include <dolfin/EmptySparsity.h>
#include <dolfin/FullSparsity.h>
#include <dolfin/MatrixSparsity.h>
#include <dolfin/TableSparsity.h>
#include <dolfin/GenericSparsity.h>
#include <dolfin/Sparsity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Sparsity::Sparsity(int N)
{
  sparsity = new FullSparsity(N);
  this->N = N;
}
//-----------------------------------------------------------------------------
Sparsity::~Sparsity()
{
  if ( sparsity )
    delete sparsity;
  sparsity = 0;
}
//-----------------------------------------------------------------------------
void Sparsity::clear()
{
  if ( sparsity )
    delete sparsity;

  sparsity = new EmptySparsity(N);
}
//-----------------------------------------------------------------------------
void Sparsity::full()
{
  if ( sparsity )
    delete sparsity;

  sparsity = new FullSparsity(N);
}
//-----------------------------------------------------------------------------
void Sparsity::setsize(int i, int size)
{
  dolfin_assert(sparsity);
  
  if ( sparsity->type() != GenericSparsity::table ) {
    delete sparsity;
    sparsity = new TableSparsity(N);
  }
  
  sparsity->setsize(i, size);
}
//-----------------------------------------------------------------------------
void Sparsity::set(int i, int j)
{
  dolfin_assert(sparsity);

  if ( sparsity->type() != GenericSparsity::table ) {
    delete sparsity;
    sparsity = new TableSparsity(N);
  }
  
  sparsity->set(i,j);
}
//-----------------------------------------------------------------------------
void Sparsity::set(const Matrix& A)
{
  if ( sparsity )
    delete sparsity;
  
  sparsity = new MatrixSparsity(N,A);
}
//-----------------------------------------------------------------------------
void Sparsity::guess(ODE& ode)
{
  if ( sparsity )
    delete sparsity;

  sparsity = new AutomaticSparsity(N, ode);
}
//-----------------------------------------------------------------------------
void Sparsity::show() const
{
  dolfin_info("Sparsity pattern:");

  for (int i = 0; i < N; i++) {
    cout << i << ":";
    for (Iterator it(i, *this); !it.end(); ++it)
      cout << " " << *it;
    cout << endl;
  }
}
//-----------------------------------------------------------------------------
GenericSparsity::Iterator* Sparsity::createIterator(int i) const
{
  dolfin_assert(sparsity);
  
  return sparsity->createIterator(i);
}
//-----------------------------------------------------------------------------
Sparsity::Iterator::Iterator(int i, const Sparsity& sparsity)
{
  iterator = sparsity.createIterator(i);
}
//-----------------------------------------------------------------------------
Sparsity::Iterator::~Iterator()
{
  if ( iterator )
    delete iterator;
  iterator = 0;
}
//-----------------------------------------------------------------------------
Sparsity::Iterator& Sparsity::Iterator::operator++()
{
  ++(*iterator);

  return *this;
}
//-----------------------------------------------------------------------------
int Sparsity::Iterator::operator*() const
{
  return **iterator;
}
//-----------------------------------------------------------------------------
Sparsity::Iterator::operator int() const
{
  return **iterator;
}
//-----------------------------------------------------------------------------
bool Sparsity::Iterator::end() const
{
  return iterator->end();
}
//-----------------------------------------------------------------------------
