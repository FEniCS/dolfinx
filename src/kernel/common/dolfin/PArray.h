// Copyright (C) 2002-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2002
// Last changed: 2005

#ifndef __P_ARRAY_H
#define __P_ARRAY_H

#include <signal.h>

#include <dolfin/dolfin_log.h>

namespace dolfin
{

  enum Range { all };
  enum Index { first, last };
  
  /// DON'T USE PArray. USE Array INSTEAD. WILL REPLACE Array IN FUTURE VERSIONS.

  /// An PArray is a list of constant size that can be used to
  /// store (often short) sets of data.
  ///
  /// PArray is constructed to use minimal storage. Only the size of the list
  /// and the elements themselves are stored.
  /// 
  /// In addition to working as a standard array (which can do only
  /// indexing), the PArray class has a couple of special purpose
  /// functions that can be used to add elements dynamically. These
  /// should be used with caution!  Consider using the List class
  /// instead. The add() functions adds a new element at the first
  /// empty position. A position is empty if for that element the
  /// operator ! returns true. An example:
  ///
  ///   PArray<Node*> nodes(5);             // Create array of length 5
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
  /// however, an PArray may be preferred even in a case where elements
  /// need to be added dynamically, for instance when a large number of
  /// such PArrays are needed (to save memory).
  ///
  /// Note that iterators don't skip empty positions. Note also that
  /// to use the PArray class in this way, the element class needs to
  /// implement the two operators
  ///
  ///   void operator= (int)   (assignment to zero)
  ///   bool operator! ()      (check if empty)
  ///
  /// These two operators work naturally for pointers.

  template <class T> class PArray
  {
  public:
    
    class Iterator;
    friend class Iterator;
    
    /// Create an empty array of size zero
    PArray();
    
    /// Create an empty array of given size
    PArray(int size);
    
    /// Destructor
    ~PArray();
    
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

    /// Iterator for the PArray class. Should be used as follows:
    ///
    /// for (PArray<T>::Iterator it(array); !it.end(); ++it) {
    ///     it->...();
    /// }

    class Iterator {
    public:
      
      /// Create an iterator positioned at the end of the array
      Iterator();
      
      /// Create an iterator positioned at the beginning of the array
      Iterator(const PArray<T>& array);

      /// Create an iterator positioned at the given position
      Iterator(const PArray<T>& array, Index index);

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
  // Implementation of PArray
  //---------------------------------------------------------------------------
  template <class T> PArray<T>::PArray()
  {
    //dolfin_debug("PArray ctor");

    array = 0;
    _size = 0;
  }
  //---------------------------------------------------------------------------    
  template <class T> PArray<T>::PArray(int size)
  {
    //dolfin_debug("PArray size ctor");

    array = 0;
    _size = 0;
    init(size);
  }
  //---------------------------------------------------------------------------    
  template <class T> PArray<T>::~PArray()
  {
    clear();
  }
  //---------------------------------------------------------------------------
  template <class T> void PArray<T>::init(int new_size)
  {
    if ( array )
      clear();
    
    if ( new_size <= 0 )
      return;
    
    array = new T[new_size];
    _size = new_size;
  }
  //---------------------------------------------------------------------------    
  template <class T> void PArray<T>::resize(int new_size)
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
  template <class T> void PArray<T>::clear()
  {
    if ( array )
      delete [] array;
    array = 0;
    _size = 0;
  }
  //---------------------------------------------------------------------------    
  template <class T> T& PArray<T>::operator() (int i) const
  {
    dolfin_assert(i >= 0);
    dolfin_assert(i < _size);
    return array[i];
  }
  //---------------------------------------------------------------------------    
  template <class T> void PArray<T>::operator= (const T& element)
  {
    for (int i = 0; i < _size; i++)
      array[i] = element;
  }
  //---------------------------------------------------------------------------    
  template <class T> int PArray<T>::size() const
  {
    return _size;
  }
  //---------------------------------------------------------------------------    
  template <class T> bool PArray<T>::empty() const
  {
    return _size == 0;
  }
  //---------------------------------------------------------------------------    
  template <class T> bool PArray<T>::contains(const T& element)
  {
    for (int i = 0; i < _size; i++)
      if ( array[i] == element )
	return true;
    return false;
  }
  //---------------------------------------------------------------------------    
  template <class T> void PArray<T>::remove(const T& element)
  {
    for (int i = 0; i < _size; i++)
      if ( array[i] == element ) {
	array[i] = 0;
	return;
      }
    dolfin_error("Element is not in the array.");
  }
  //---------------------------------------------------------------------------	 
  template <class T> T& PArray<T>::max() const
  {
    int pos = 0;
    for (int i = 1; i < _size; i++)
      if ( array[pos] < array[i] )
	pos = i;
    
    return array[pos];
  }
  //---------------------------------------------------------------------------
  template <class T> void PArray<T>::swap(int i, int j)
  {
    T tmp = array[i];
    array[i] = array[j];
    array[j] = tmp;
  }
  //---------------------------------------------------------------------------	 
  template <class T> typename PArray<T>::Iterator PArray<T>::begin() const
    {
    return Iterator(*this);
  }
  //---------------------------------------------------------------------------    
  template <class T> void PArray<T>::setsize(int new_size)
  {
    _size = new_size;
  }
  //---------------------------------------------------------------------------    
  template <class T> void PArray<T>::reset()
  {
    for (int i = 0; i < _size; i++)
      array[i] = 0;
  }
  //---------------------------------------------------------------------------    
  template <class T> void PArray<T>::init()
  {
    init(_size);
    reset();
  }
  //---------------------------------------------------------------------------    
  template <class T> int PArray<T>::add(T element)
  {
    for (int i = 0; i < _size; i++)
      if ( !array[i] ) {
	array[i] = element;
	return i;
      }

    dolfin_segfault();
    dolfin_error("PArray is full.");
    return -1;
  }
  //--------------------------------------------------------------------------- 
  template <class T> void PArray<T>::resize()
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
  // Implementatio of PArray::Iterator
  //---------------------------------------------------------------------------
  template <class T> PArray<T>::Iterator::Iterator()
  {
    array = 0;
    element = 0;
    _index = 0;
    size = 0;
    at_end = true;
  }
  //---------------------------------------------------------------------------      
  template <class T> PArray<T>::Iterator::Iterator(const PArray<T> &array)
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
  template <class T> PArray<T>::Iterator::Iterator
  (const PArray<T> &array, Index index)
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
  template <class T> typename PArray<T>::Iterator::Iterator& 
  PArray<T>::Iterator::operator++()
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
  template <class T> typename PArray<T>::Iterator::Iterator& 
  PArray<T>::Iterator::operator--()
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
  template <class T> bool PArray<T>::Iterator::end() const
  {
    return at_end;
  }
  //---------------------------------------------------------------------------      
  template <class T> bool PArray<T>::Iterator::last() const
  {
    return _index == (size - 1);
  }
  //---------------------------------------------------------------------------
  template <class T> bool PArray<T>::Iterator::operator==(const Iterator& it)
  {
    return element == it.element;
  }
  //---------------------------------------------------------------------------      
  template <class T> void PArray<T>::Iterator::operator=(Iterator& it)
  {
    array   = it.array;
    element = it.element;
    _index  = it._index;
    size    = it.size;
    at_end  = it.at_end;
  }
  //---------------------------------------------------------------------------
  template <class T> void PArray<T>::Iterator::operator=(const Iterator& it)
  {
    array   = it.array;
    element = it.element;
    _index  = it._index;
    size    = it.size;
    at_end  = it.at_end;
  }
  //---------------------------------------------------------------------------
  template <class T> int PArray<T>::Iterator::index() const
  {
    return _index;
  }
  //---------------------------------------------------------------------------      
  template <class T> T& PArray<T>::Iterator::operator*() const
  {
    return *element;
  }
  //---------------------------------------------------------------------------      
  template <class T> T* PArray<T>::Iterator::operator->() const
  {
    return element;
  }
  //---------------------------------------------------------------------------      
  template <class T> PArray<T>::Iterator::operator T*() const
  {
    return element;
  }
  //---------------------------------------------------------------------------      
  template <class T> T* PArray<T>::Iterator::pointer() const
  {
    return element;
  }
  //---------------------------------------------------------------------------      
  
}

#endif
