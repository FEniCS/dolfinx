// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/NewArray.h>
#include <dolfin/AutomaticSparsity.h>
#include <dolfin/EmptySparsity.h>
#include <dolfin/FullSparsity.h>
#include <dolfin/MatrixSparsity.h>
#include <dolfin/TableSparsity.h>
#include <dolfin/GenericSparsity.h>
#include <dolfin/Sparsity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Sparsity::Sparsity(unsigned int N)
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
void Sparsity::setsize(unsigned int i, unsigned int size)
{
  dolfin_assert(sparsity);
  
  if ( sparsity->type() != GenericSparsity::table ) {
    delete sparsity;
    sparsity = new TableSparsity(N);
  }
  
  sparsity->setsize(i, size);
}
//-----------------------------------------------------------------------------
void Sparsity::set(unsigned int i, unsigned int j)
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
void Sparsity::transp(const Sparsity& sparsity)
{
  // Clear old sparsity
  if ( this->sparsity )
    delete this->sparsity;

  // Get size of system
  N = sparsity.N;

  // If sparsity is full, then the transpose is full
  if ( sparsity.sparsity->type() == GenericSparsity::full )
  {
    this->sparsity = new FullSparsity(N);
    return;
  }

  // If sparsity is empty, then the transpose is empty
  if ( sparsity.sparsity->type() == GenericSparsity::empty )
  {
    this->sparsity = new EmptySparsity(N);
    return;
  }

  // Otherwise, create table sparsity
  this->sparsity = new TableSparsity(N);

  // Count the number of dependencies
  NewArray<unsigned int> rowsizes(N);
  rowsizes = 0;
  for (unsigned int i = 0; i < N; i++)
    for (Iterator it(i, sparsity); !it.end(); ++it)
      rowsizes[*it]++;

  // Set row sizes
  for (unsigned int i = 0; i < N; i++)
    this->sparsity->setsize(i, rowsizes[i]);

  // Set dependencies
  for (unsigned int i = 0; i < N; i++)
    for (Iterator it(i, sparsity); !it.end(); ++it)
      this->sparsity->set(*it, i);
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

  for (unsigned int i = 0; i < N; i++) {
    cout << i << ":";
    for (Iterator it(i, *this); !it.end(); ++it)
      cout << " " << *it;
    cout << endl;
  }
}
//-----------------------------------------------------------------------------
GenericSparsity::Iterator* Sparsity::createIterator(unsigned int i) const
{
  dolfin_assert(sparsity);
  
  return sparsity->createIterator(i);
}
//-----------------------------------------------------------------------------
Sparsity::Iterator::Iterator(unsigned int i, const Sparsity& sparsity)
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
unsigned int Sparsity::Iterator::operator*() const
{
  return **iterator;
}
//-----------------------------------------------------------------------------
Sparsity::Iterator::operator unsigned int() const
{
  return **iterator;
}
//-----------------------------------------------------------------------------
bool Sparsity::Iterator::end() const
{
  return iterator->end();
}
//-----------------------------------------------------------------------------
