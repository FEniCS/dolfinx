// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SHORT_LIST_H
#define __SHORT_LIST_H

#include <dolfin/dolfin_log.h>

namespace dolfin {

  /// ShortList implements a list of given constant size.
  ///
  /// Memory usage:      Only the elements and the size of the list are stored
  /// Adding elements:   The add() function uses a linear search to find the next position
  /// Changing the size: Use the resize() function to change the size of the list.

  
  template <class T> class ShortList {
  public:
    
    class Iterator;
    friend class Iterator;
    
    /// Create an empty list of size zero
    ShortList();
    
    /// Create an empty list of given size
    ShortList(int size);
    
    /// Destructor
    ~ShortList();
    
    /// Initialise list to given size
    void init(int new_size);
    
    /// Resize to a list of given size (keeping old elements)
    void resize(int new_size);

    /// Clear list
    void clear();
    
    /// Indexing
    T& operator() (int i) const;

    /// Set all elements equal to given element
    void operator= (const T& element);
   
    /// Return size of list
    int size() const;
        
    /// Search list for given element
    bool contains(T element);
    
    /// Remove given element (first one matching)
    void remove(T element);

    /// Swap two elements
    void swap(int i, int j);
	 
    /// Return an iterator to the beginning of the list
    Iterator begin() const;

    /// --- Special functions ---
    
    /// Set size of list (useful in combination with init() or resize())
    void setsize(int new_size);
    
    /// Initialize list to previously specified size and set all elements = 0
    void init();
   
    /// Add element to next available position
    int add(T element);

    /// Remove empty elements
    void resize();

    /// Iterator for the List class. Should be used as follows:
    ///
    ///     for (ShortList<T>::Iterator it(list); !it.end(); ++it) {
    ///         it->...();
    ///     }

    class Iterator {
    public:
      
      /// Create an iterator positioned at the end of the list
      Iterator();
      
      /// Create an iterator positioned at the beginning of the list
      Iterator(const ShortList<T> &list);

      Iterator& operator++();

      bool end() const;
      
      bool last() const;

      bool operator==(const Iterator& it);
      
      void operator=(Iterator& it);

      void operator=(const Iterator& it);

      int index() const;
      
      T& operator*() const;
      
      T* operator->() const;
      
      T* pointer() const;
      
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

  //---------------------------------------------------------------------------
  // Implementation of ShortList
  //---------------------------------------------------------------------------
  template <class T> ShortList<T>::ShortList()
  {
    list = 0;
    _size = 0;
  }
  //---------------------------------------------------------------------------    
  template <class T> ShortList<T>::ShortList(int size)
  {
    list = 0;
    _size = 0;
    init(size);
  }
  //---------------------------------------------------------------------------    
  template <class T> ShortList<T>::~ShortList()
  {
    clear();
  }
  //---------------------------------------------------------------------------
  template <class T> void ShortList<T>::init(int new_size)
  {
    if ( list )
      clear();
    
    if ( new_size <= 0 )
      return;
    
    list = new T[new_size];
    _size = new_size;
  }
  //---------------------------------------------------------------------------    
  template <class T> void ShortList<T>::resize(int new_size)
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
  //---------------------------------------------------------------------------    
  template <class T> void ShortList<T>::clear()
  {
    if ( list )
      delete [] list;
    list = 0;
    _size = 0;
  }
  //---------------------------------------------------------------------------    
  template <class T> T& ShortList<T>::operator() (int i) const
  {
    return list[i];
  }
  //---------------------------------------------------------------------------    
  template <class T> void ShortList<T>::operator= (const T& element)
  {
    for (int i = 0; i < _size; i++)
      list[i] = element;
  }
  //---------------------------------------------------------------------------    
  template <class T> int ShortList<T>::size() const
  {
    return _size;
  }
  //---------------------------------------------------------------------------    
  template <class T> bool ShortList<T>::contains(T element)
  {
    for (int i = 0; i < _size; i++)
      if ( list[i] == element )
	return true;
    return false;
  }
  //---------------------------------------------------------------------------    
  template <class T> void ShortList<T>::remove(T element)
  {
    for (int i = 0; i < _size; i++)
      if ( list[i] == element ) {
	list[i] = 0;
	return;
      }
    dolfin_error("Element is not in the list.");
  }
  //---------------------------------------------------------------------------
  template <class T> void ShortList<T>::swap(int i, int j)
  {
    T tmp = list[i];
    list[i] = list[j];
    list[j] = tmp;
  }
  //---------------------------------------------------------------------------	 
  template <class T> ShortList<T>::Iterator ShortList<T>::begin() const
  {
    return Iterator(*this);
  }
  //---------------------------------------------------------------------------    
  template <class T> void ShortList<T>::setsize(int new_size)
  {
    _size = new_size;
  }
  //---------------------------------------------------------------------------    
  template <class T> void ShortList<T>::init()
  {
    init(_size);

    for (int i = 0; i < _size; i++)
      list[i] = 0; // Requires T::operator=(int i)
  }
  //---------------------------------------------------------------------------    
  template <class T> int ShortList<T>::add(T element)
  {
    for (int i = 0; i < _size; i++)
      if ( !list[i] ) {
	list[i] = element;
	return i;
      }
    return -1;
  }
  //--------------------------------------------------------------------------- 
  template <class T> void ShortList<T>::resize()
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
  //---------------------------------------------------------------------------
  // Implementatio of ShortList::Iterator
  //---------------------------------------------------------------------------
  template <class T> ShortList<T>::Iterator::Iterator()
  {
    list = 0;
    element = 0;
    _index = 0;
    size = 0;
    at_end = true;
  }
  //---------------------------------------------------------------------------      
  template <class T> ShortList<T>::Iterator::Iterator(const ShortList<T> &list)
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
  //---------------------------------------------------------------------------      
  template <class T> ShortList<T>::Iterator::Iterator& 
  ShortList<T>::Iterator::operator++()
  {
    if ( _index == (size - 1) )
      at_end = true;
    else
      _index++;
    
    element = list + _index;
    
    return *this;
  }
  //---------------------------------------------------------------------------      
  template <class T> bool ShortList<T>::Iterator::end() const
  {
    return at_end;
  }
  //---------------------------------------------------------------------------      
  template <class T> bool ShortList<T>::Iterator::last() const
  {
    return _index == (size - 1);
  }
  //---------------------------------------------------------------------------
  template <class T> bool ShortList<T>::Iterator::operator==(const Iterator& it)
  {
    return element == it.element;
  }
  //---------------------------------------------------------------------------      
  template <class T> void ShortList<T>::Iterator::operator=(Iterator& it)
  {
    list    = it.list;
    element = it.element;
    _index  = it._index;
    size    = it.size;
    at_end  = it.at_end;
  }
  //---------------------------------------------------------------------------
  template <class T> void ShortList<T>::Iterator::operator=(const Iterator& it)
  {
    list    = it.list;
    element = it.element;
    _index  = it._index;
    size    = it.size;
    at_end  = it.at_end;
  }
  //---------------------------------------------------------------------------
  template <class T> int ShortList<T>::Iterator::index() const
  {
    return _index;
  }
  //---------------------------------------------------------------------------      
  template <class T> T& ShortList<T>::Iterator::operator*() const
  {
    return *element;
  }
  //---------------------------------------------------------------------------      
  template <class T> T* ShortList<T>::Iterator::operator->() const
  {
    return element;
  }
  //---------------------------------------------------------------------------      
  template <class T> T* ShortList<T>::Iterator::pointer() const
  {
    return element;
  }
  //---------------------------------------------------------------------------      
  
}

#endif
