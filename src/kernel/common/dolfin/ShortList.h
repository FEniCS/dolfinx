// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SHORT_LIST_H
#define __SHORT_LIST_H

template <class T> class ShortList {
public:

  class Iterator;
  friend class Iterator;
  
  /// Creates an empty list
  ShortList()
  {
	 list = 0;
	 _size = 0;
  }

  /// Creates a zero list of given size
  ShortList(int new_size)
  {
	 list = 0;
	 _size = 0;
	 init(new_size);
  }

  /// The destructor
  ~ShortList()
  {
	 clear();
  }

  /// Indexing
  T& operator() (int i)
  {
	 return list[i];
  }
  
  /// Returns the size of the list. Note that size can be changed.
  /// This can be useful in combination with init() and resize().
  int& size()
  {
	 return _size;
  }

  /// Adds the element to the next available position
  void add(T element)
  {
	 for (int i = 0; i < _size; i++)
		if ( !list[i] ){
		  list[i] = element;
		  return;
		}
  }
  
  /// Initialises the list to a zero list of given size
  void init(int new_size)
  {
	 if ( list )
		clear();
	 
	 if ( new_size <= 0 )
		return;
	 
	 list = new T[new_size];
	 for (int i = 0; i < new_size; i++)
		list[i] = 0;
	 _size = new_size;
	 
  }

  /// Initialises the list to zero list of the current size
  void init()
  {
	 init(_size);
  }

  /// Removes unused (zero) elements
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

  /// Searches the list for the element
  bool contains(T element)
  {
	 for (int i = 0; i < _size; i++)
		if ( list[i] == element )
		  return true;
	 return false;
  }

  /// Clears the list
  void clear()
  {
	 if ( list )
		delete [] list;
	 list = 0;
	 _size = 0;
  }
  
  /// Returns an iterator to the beginning of the list
  Iterator begin()
  {
	 return Iterator(*this);
  }
  
  /// Iterator
  class Iterator {
  public:

	 /// Creates an iterator positioned at the end of the list
	 Iterator()
	 {
		list = 0;
		element = 0;
		index = 0;
		size = 0;
		at_end = true;
	 }
	 
	 /// Creates an iterator positioned at the beginning of the list
	 Iterator(ShortList<T> &list)
	 {
		if ( list._size > 0 ){
		  element = list.list;
		  at_end = false;
		}
		else{
		  element = 0;
		  at_end = true;
		}

		index = 0;
		size = list._size;
		this->list = list.list;
	 }

	 Iterator& operator++()
	 {
		if ( index == (size - 1) )
		  at_end = true;
		else
		  index++;
		
		element = list + index;
		
		return *this;
	 }

	 bool end()
	 {
		return at_end;
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
	 int index;
	 int size;
	 bool at_end;
	 
  };

private:

  T *list;
  int _size;
    
};

#endif
