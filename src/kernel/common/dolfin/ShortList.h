// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SHORT_LIST_H
#define __SHORT_LIST_H

// ShortList implements a list of given constant size.
//
// Memory usage:      Only the elements and the size of the list are stored
// Adding elements:   The add() function uses a linear search to find the next position
// Changing the size: Use the resize() function to change the size of the list.
//
// To use this template with a given class T, assignment to zero (0) must be
// possible for objects of class T. This is used to mark empty positions in
// the list. Typically the list is used with pointers, where assignment to
// zero naturally means that an object is empty.
//
// To use the full functionality of this template, the operator ! is also
// needed for objects of class T. When ! is true, this should indicate that
// the object is empty, which should also correspond to assignment = 0.

namespace dolfin {

  template <class T> class ShortList {
  public:
	 
	 class Iterator;
	 friend class Iterator;
	 
	 /// Create an empty list of size zero
	 ShortList()
	 {
		list = 0;
		_size = 0;
	 }
	 
	 /// Create an empty list of given size
	 ShortList(int size)
	 {
		list = 0;
		_size = 0;
		init(size);
	 }
	 
	 /// Destructor
	 ~ShortList()
	 {
		clear();
	 }

	 /// Initialise to an empty list of given size
	 void init(int new_size)
	 {
		if ( list )
		  clear();
		
		if ( new_size <= 0 )
		  return;
		
		list = new T[new_size];
		for (int i = 0; i < new_size; i++)
		  list[i] = 0; // Requires = 0 for class T
		_size = new_size;
	 }
	 
	 /// Initialise to an empty list of current size
	 void init()
	 {
		init(_size);
	 }
	 
	 /// Remove empty elements
	 void resize()
	 {
		if ( !list )
		  return;
		
		// Count the number of used positions
		int new_size = 0;
		for (int i = 0; i < _size; i++)
		  if ( list[i] )
			 new_size++;
		
		if ( new_size == 0 ){
		  clear();
		  return;
		}
		
		// Copy and reallocate
		T *new_list = new T[new_size];
		int pos = 0;
		for (int i = 0; i < _size; i++)
		  if ( list[i] )
			 new_list[pos++] = list[i];
		delete [] list;
		list = new_list;
		_size = new_size;
	 }
	 
	 /// Resize to a list of given size (keeping old elements)
	 void resize(int new_size)
	 {
		if ( !list ) {
		  init(new_size);
		  return;
		}
		
		// Create new list and copy the elements
		T *new_list = new T[new_size];
		for (int i = 0; i < _size && i < new_size; i++)
		  new_list[i] = list[i];
		
		// Update the old list with the new list
		delete [] list;
		list = new_list;
		_size = new_size;
	 }

	 /// Clear list
	 void clear()
	 {
		if ( list )
		  delete [] list;
		list = 0;
		_size = 0;
	 }
	 
	 /// Indexing
	 T& operator() (int i) const
	 {
		return list[i];
	 }
	 
	 /// Get size of list
	 int size() const
	 {
		return _size;
	 }
	 
	 /// Set size of list (useful in combination with init() or resize())
	 int setsize(int new_size)
	 {
		_size = new_size;
	 }
	 
	 /// Add element to next available position
	 int add(T element)
	 {
		for (int i = 0; i < _size; i++)
		  if ( !list[i] ){
			 list[i] = element;
			 return i;
		  }
		return -1;
	 }
	 
	 /// Search list for given element
	 bool contains(T element)
	 {
		for (int i = 0; i < _size; i++)
		  if ( list[i] == element )
			 return true;
		return false;
	 }
	 
	 /// Return an iterator to the beginning of the list
	 Iterator begin() const
	 {
		return Iterator(*this);
	 }
	 
	 /// Iterator
	 class Iterator {
	 public:
		
		/// Create an iterator positioned at the end of the list
		Iterator()
		{
		  list = 0;
		  element = 0;
		  _index = 0;
		  size = 0;
		  at_end = true;
		}
		
		/// Create an iterator positioned at the beginning of the list
		Iterator(const ShortList<T> &list)
		{
		  if ( list._size > 0 ){
			 element = list.list;
			 at_end = false;
		  }
		  else{
			 element = 0;
			 at_end = true;
		  }
		  
		  _index = 0;
		  size = list._size;
		  this->list = list.list;
		}
		
		Iterator& operator++()
		{
		  if ( _index == (size - 1) )
			 at_end = true;
		  else
			 _index++;
		  
		  element = list + _index;
		  
		  return *this;
		}
		
		bool end() const
		{
		  return at_end;
		}
		
		int index() const
		{
		  return _index;
		}
		
		T& operator*() const
		{
		  return *element;
		}
		
		T* operator->() const
		{
		  return element;
		}
		
		T* pointer() const
		{
		  return element;
		}
		
	 private:
		
		T *list;
		T *element;
		int _index;
		int size;
		bool at_end;
		
	 };
	 
  private:
	 
	 T *list;
	 int _size;
    
  };

}

#endif
