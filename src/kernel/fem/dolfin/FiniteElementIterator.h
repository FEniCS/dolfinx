// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FINITE_ELEMENT_ITERATOR_H
#define __FINITE_ELEMENT_ITERATOR_H

namespace dolfin {

  typedef FiniteElement* FiniteElementPointer;
  
  class FiniteElementIterator {
  public:
	 
	 FiniteElementIterator(const Grid &grid);
	 
	 bool end() const;
	 int  index() const;

	 FiniteElementIterator& operator++();
	 
	 operator FiniteElement&() const;
	 operator FiniteElementPointer() const;
	 
	 FiniteElement& operator*() const;
	 FiniteElement* operator->() const;

  private:

	 CellIterator* c;

  };

}
  
#endif
