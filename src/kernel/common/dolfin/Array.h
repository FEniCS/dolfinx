// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ARRAY_H
#define __ARRAY_H

#include <signal.h>

#include <dolfin/dolfin_log.h>
#include <dolfin/General.h>

namespace dolfin {
  
  /// DON'T USE ARRAY. USE NEWARRAY INSTEAD. WILL REPLACE ARRAY IN FUTURE VERSIONS.

  /// An Array is a list of constant size that can be used to
  /// store (often short) sets of data.
  ///
  /// Array is constructed to use minimal storage. Only the size of the list
  /// and the elements themselves are stored.
  /// 
  /// In addition to working as a standard array (which can do only
  /// indexing), the Array class has a couple of special purpose
  /// functions that can be used to add elements dynamically. These
  /// should be used with caution!  Consider using the List class
  /// instead. The add() functions adds a new element at the first
  /// empty position. A position is empty if for that element the
  /// operator ! returns true. An example:
  ///
  ///   Array<Node*> nodes(5);             // Create array of length 5
  ///   nodes.reset();                     // Assign 0 to all pointers
  ///   while ( ... ) {                    //
  ///     if ( nodes.add(node) != -1 ) {   // Add node and check if full
  ///       nodes.resize(2*nodes.size());  // If full, resize ...
  ///       nodes.add(node);               // ... and try again
  ///     }                                //
  ///   }                                  //
  ///
  /// If you want a list to which you can dynamically add elements, you
  /// should probably use the List class (for reasonably small sets of
  /// data) or the Table class (for large sets of data). Sometimes,
  /// however, an Array may be preferred even in a case where elements
  /// need to be added dynamically, for instance when a large number of
  /// such Arrays are needed (to save memory).
  ///
  /// Note that iterators don't skip empty positions. Note also that
  /// to use the Array class in this way, the element class needs to
  /// implement the two operators
  ///
  ///   void operator= (int)   (assignment to zero)
  ///   bool operator! ()      (check if empty)
  ///
  /// These two operators work naturally for pointers.

  template <class T> class Array {
  public:
    
    class Iterator;
    friend class Iterator;
    
    /// Create an empty array of size zero
    Array();
    
    /// Create an empty array of given size
    Array(int size);
    
    /// Destructor
    ~Array();
    
    /// Initialise array to given size
    void init(int new_size);
    
    /// Resize to an array of given size (keeping old elements)
    void resize(int new_size);

    /// Clear array
    void clear();
    
    /// Indexing
    T& operator() (int i) const;

    /// Set all elements equal to given element
    void operator= (const T& element);
   
    /// Return size of array
    int size() const;

    /// Check if array is empty
    bool empty() const;
        
    /// Check if the array contains a given element
    bool contains(const T& element);
    
    /// Return the maximum element
    T& max() const;

    /// Remove given element (first one matching)
    void remove(const T& element);

    /// Swap two elements
    void swap(int i, int j);
	 
    /// Return an iterator to the beginning of the array
    Iterator begin() const;

    /// --- Special functions ---

    /// Assign 0 to all elements
    void reset();
    
    /// Set size of array (useful in combination with init() or resize())
    void setsize(int new_size);
    
    /// Initialize array to previously specified size and assign 0 to all elements
    void init();
   
    /// Add element to next available position
    int add(T element);

    /// Remove empty elements
    void resize();

    /// Iterator for the Array class. Should be used as follows:
    ///
    /// for (Array<T>::Iterator it(array); !it.end(); ++it) {
    ///     it->...();
    /// }

    class Iterator {
    public:
      
      /// Create an iterator positioned at the end of the array
      Iterator();
      
      /// Create an iterator positioned at the beginning of the array
      Iterator(const Array<T>& array);

      /// Create an iterator positioned at the given position
      Iterator(const Array<T>& array, Index index);

      Iterator& operator++();

      Iterator& operator--();

      bool end() const;
      
      bool last() const;

      bool operator==(const Iterator& it);
      
      void operator=(Iterator& it);

      void operator=(const Iterator& it);

      int index() const;
      
      T& operator*() const;
      
      T* operator->() const;
      
      operator T*() const;

      T* pointer() const;
      
    private:
      
      T *array;
      T *element;
      int _index;
      int size;
      bool at_end;
      
    };
    
  private:
    
    T *array;
    int _size;
    
  };

  //---------------------------------------------------------------------------
  // Implementation of Array
  //---------------------------------------------------------------------------
  template <class T> Array<T>::Array()
  {
    //dolfin_debug("Array ctor");

    array = 0;
    _size = 0;
  }
  //---------------------------------------------------------------------------    
  template <class T> Array<T>::Array(int size)
  {
    //dolfin_debug("Array size ctor");

    array = 0;
    _size = 0;
    init(size);
  }
  //---------------------------------------------------------------------------    
  template <class T> Array<T>::~Array()
  {
    clear();
  }
  //---------------------------------------------------------------------------
  template <class T> void Array<T>::init(int new_size)
  {
    if ( array )
      clear();
    
    if ( new_size <= 0 )
      return;
    
    array = new T[new_size];
    _size = new_size;
  }
  //---------------------------------------------------------------------------    
  template <class T> void Array<T>::resize(int new_size)
  {
    if ( !array ) {
      init(new_size);
      return;
    }
    
    // Create new array and copy the elements
    T *new_array = new T[new_size];
    for (int i = 0; i < _size && i < new_size; i++)
      new_array[i] = array[i];
    
    // Update the old array with the new array
    delete [] array;
    array = new_array;
    _size = new_size;
  }
  //---------------------------------------------------------------------------    
  template <class T> void Array<T>::clear()
  {
    if ( array )
      delete [] array;
    array = 0;
    _size = 0;
  }
  //---------------------------------------------------------------------------    
  template <class T> T& Array<T>::operator() (int i) const
  {
    dolfin_assert(i >= 0);
    dolfin_assert(i < _size);
    return array[i];
  }
  //---------------------------------------------------------------------------    
  template <class T> void Array<T>::operator= (const T& element)
  {
    for (int i = 0; i < _size; i++)
      array[i] = element;
  }
  //---------------------------------------------------------------------------    
  template <class T> int Array<T>::size() const
  {
    return _size;
  }
  //---------------------------------------------------------------------------    
  template <class T> bool Array<T>::empty() const
  {
    return _size == 0;
  }
  //---------------------------------------------------------------------------    
  template <class T> bool Array<T>::contains(const T& element)
  {
    for (int i = 0; i < _size; i++)
      if ( array[i] == element )
	return true;
    return false;
  }
  //---------------------------------------------------------------------------    
  template <class T> void Array<T>::remove(const T& element)
  {
    for (int i = 0; i < _size; i++)
      if ( array[i] == element ) {
	array[i] = 0;
	return;
      }
    dolfin_error("Element is not in the array.");
  }
  //---------------------------------------------------------------------------	 
  template <class T> T& Array<T>::max() const
  {
    int pos = 0;
    for (int i = 1; i < _size; i++)
      if ( array[pos] < array[i] )
	pos = i;
    
    return array[pos];
  }
  //---------------------------------------------------------------------------
  template <class T> void Array<T>::swap(int i, int j)
  {
    T tmp = array[i];
    array[i] = array[j];
    array[j] = tmp;
  }
  //---------------------------------------------------------------------------	 
  template <class T> typename Array<T>::Iterator Array<T>::begin() const
    {
    return Iterator(*this);
  }
  //---------------------------------------------------------------------------    
  template <class T> void Array<T>::setsize(int new_size)
  {
    _size = new_size;
  }
  //---------------------------------------------------------------------------    
  template <class T> void Array<T>::reset()
  {
    for (int i = 0; i < _size; i++)
      array[i] = 0;
  }
  //---------------------------------------------------------------------------    
  template <class T> void Array<T>::init()
  {
    init(_size);
    reset();
  }
  //---------------------------------------------------------------------------    
  template <class T> int Array<T>::add(T element)
  {
    for (int i = 0; i < _size; i++)
      if ( !array[i] ) {
	array[i] = element;
	return i;
      }

    dolfin_debug("foo");
    raise(SIGSEGV);
    dolfin_error("Array is full.");
    dolfin_debug("foo2");
    //dolfin_segfault();
    return -1;
  }
  //--------------------------------------------------------------------------- 
  template <class T> void Array<T>::resize()
  {
    if ( !array )
      return;
    
    // Count the number of used positions
    int new_size = 0;
    for (int i = 0; i < _size; i++)
      if ( array[i] )
	new_size++;
    
    if ( new_size == 0 ){
      clear();
      return;
    }
    
    // Copy and reallocate
    T *new_array = new T[new_size];
    int pos = 0;
    for (int i = 0; i < _size; i++)
      if ( array[i] )
	new_array[pos++] = array[i];
    delete [] array;
    array = new_array;
    _size = new_size;
  }
  //---------------------------------------------------------------------------
  // Implementatio of Array::Iterator
  //---------------------------------------------------------------------------
  template <class T> Array<T>::Iterator::Iterator()
  {
    array = 0;
    element = 0;
    _index = 0;
    size = 0;
    at_end = true;
  }
  //---------------------------------------------------------------------------      
  template <class T> Array<T>::Iterator::Iterator(const Array<T> &array)
  {
    if ( array._size > 0 ){
      element = array.array;
      at_end = false;
    }
    else{
      element = 0;
      at_end = true;
    }
    
    _index = 0;
    size = array._size;
    this->array = array.array;
  }
  //---------------------------------------------------------------------------      
  template <class T> Array<T>::Iterator::Iterator
  (const Array<T> &array, Index index)
  {
    switch (index) {
    case dolfin::first:
      
      if ( array._size > 0 ){
	element = array.array;
	at_end = false;
      }
      else{
	element = 0;
	at_end = true;
      }
      
      _index = 0;
      size = array._size;
      this->array = array.array;
      
      break;

    case dolfin::last:
      
      if ( array._size > 0 ){
	element = array.array + array._size - 1;
	at_end = false;
      }
      else{
	element = 0;
	at_end = true;
      }
      
      _index = array._size - 1;
      size = array._size;
      this->array = array.array;

      break;
      
    default:
      
      dolfin_error("Unknown iterator position.");

    }      

  }
  //---------------------------------------------------------------------------      
  template <class T> typename Array<T>::Iterator::Iterator& 
  Array<T>::Iterator::operator++()
  {
    if ( _index == (size - 1) )
      at_end = true;
    else {
      at_end = false;
      _index++;
    }
    
    element = array + _index;
    
    return *this;
  }
  //---------------------------------------------------------------------------      
  template <class T> typename Array<T>::Iterator::Iterator& 
  Array<T>::Iterator::operator--()
  {
    if ( _index == 0 )
      at_end = true;
    else {
      at_end = false;
      _index--;
    }
    
    element = array + _index;
    
    return *this;
  }
  //---------------------------------------------------------------------------      
  template <class T> bool Array<T>::Iterator::end() const
  {
    return at_end;
  }
  //---------------------------------------------------------------------------      
  template <class T> bool Array<T>::Iterator::last() const
  {
    return _index == (size - 1);
  }
  //---------------------------------------------------------------------------
  template <class T> bool Array<T>::Iterator::operator==(const Iterator& it)
  {
    return element == it.element;
  }
  //---------------------------------------------------------------------------      
  template <class T> void Array<T>::Iterator::operator=(Iterator& it)
  {
    array   = it.array;
    element = it.element;
    _index  = it._index;
    size    = it.size;
    at_end  = it.at_end;
  }
  //---------------------------------------------------------------------------
  template <class T> void Array<T>::Iterator::operator=(const Iterator& it)
  {
    array   = it.array;
    element = it.element;
    _index  = it._index;
    size    = it.size;
    at_end  = it.at_end;
  }
  //---------------------------------------------------------------------------
  template <class T> int Array<T>::Iterator::index() const
  {
    return _index;
  }
  //---------------------------------------------------------------------------      
  template <class T> T& Array<T>::Iterator::operator*() const
  {
    return *element;
  }
  //---------------------------------------------------------------------------      
  template <class T> T* Array<T>::Iterator::operator->() const
  {
    return element;
  }
  //---------------------------------------------------------------------------      
  template <class T> Array<T>::Iterator::operator T*() const
  {
    return element;
  }
  //---------------------------------------------------------------------------      
  template <class T> T* Array<T>::Iterator::pointer() const
  {
    return element;
  }
  //---------------------------------------------------------------------------      
  
}

#endif
