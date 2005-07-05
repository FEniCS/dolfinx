// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003
// Last changed: 2005

#ifndef __P_LIST_H
#define __P_LIST_H

#include <list>
#include <iterator>
#include <dolfin/dolfin_log.h>

namespace dolfin
{

  /// DON'T USE PList. USE List INSTEAD. WILL REPLACE PList IN FUTURE VERSIONS.

  /// A PList is used to store variable-sized sets of data and is implemented
  /// using a double linked list (std::list).

  template <class T> class PList
  {
  public:

    /// Constructor
    PList();
    
    /// Destructor
    ~PList();
    
    /// Clear list
    void clear();

    /// Add element to the list
    void add(const T& element);
    
    /// Return size of list
    int size() const;

    /// Check if list is empty
    bool empty() const;

    /// Check if the list contains a given element
    bool contains(const T& element);

    /// Remove given element (first one matching)
    void remove(const T& element);

    /// Get first element in the list and remove it
    T pop();
    
    /// Iterator for the PList class. Should be used as follows:
    ///
    /// for (PList<T>::Iterator it(list); !it.end(); ++it) {
    ///     it->...();
    /// }
    
    class Iterator {
    public:
      
      /// Create iterator positioned at the beginning of the list
      Iterator(PList<T>& list);

      /// Copy constructor
      Iterator(const Iterator& it);

      /// Current position in the list
      int index() const;
		
      /// Prefix increment
      Iterator& operator++();
		
      /// Check if iterator has reached end of the list
      bool end() const;
	
      /// Check if iterator has reached the last element
      bool last() const;

      /// Return current element
      T& operator*() const;

      /// Pointer mechanism
      T* operator->() const;

      /// Cast to pointer to current element
      operator T*() const;
		
      /// Returns pointer to current element
      T* pointer() const;
      
      /// Assignment
      void operator=(const Iterator& it);

      /// Equality operator
      bool operator==(const Iterator& it) const;
		
      /// Inequality operator
      bool operator!=(const Iterator& it) const;

    private:

      typename std::list<T>::iterator it;
      std::list<T>& l;
      int _index;

    };

    // Friends
    friend class Iterator;

  private:

    std::list<T> l;

  };

  //---------------------------------------------------------------------------
  // Implementation of PList
  //---------------------------------------------------------------------------
  template <class T> PList<T>::PList()
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <class T> PList<T>::~PList()
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <class T> void PList<T>::clear()
  {
    l.clear();    
  }
  //---------------------------------------------------------------------------
  template <class T> void PList<T>::add(const T& element)
  {
    l.push_back(element);
  }
  //---------------------------------------------------------------------------
  template <class T> int PList<T>::size() const
  {
    // Note the following from the stl manual: You should not assume
    // that this function is constant time. It is permitted to be
    // O(N), where N is the number of elements in the list.
    
    return l.size();
  }
  //---------------------------------------------------------------------------
  template <class T> bool PList<T>::empty() const
  {
    return l.empty();
  }
  //---------------------------------------------------------------------------
  template <class T> bool PList<T>::contains(const T& element)
  {
    for (typename std::list<T>::iterator it = l.begin(); it != l.end(); ++it)
      if ( *it == element )
	return true;

    return false;
  }
  //---------------------------------------------------------------------------
  template <class T> void PList<T>::remove(const T& element)
  {
    for (typename std::list<T>::iterator it = l.begin(); it != l.end(); ++it)
      if ( *it == element ) {
	l.erase(it);
	return;
      }
    dolfin_error("Element is not in the list.");
  }
  //---------------------------------------------------------------------------
  template <class T> T PList<T>::pop()
  {
    T first = l.front();
    l.pop_front();
    return first;
  }
  //---------------------------------------------------------------------------
  // Implementation of PList::Iterator
  //---------------------------------------------------------------------------
  template <class T> PList<T>::Iterator::Iterator (PList<T>& list) : l(list.l)
  {
    it = l.begin();
    _index = 0;
  }
  //---------------------------------------------------------------------------
  template <class T> PList<T>::Iterator::Iterator(const Iterator& it) : l(it.l)
  {
    this->it = it.it;
    this->_index = it._index;
  }
  //---------------------------------------------------------------------------
  template <class T> int PList<T>::Iterator::index() const
  {
    return _index;
  }
  //---------------------------------------------------------------------------
  template <class T> typename PList<T>::Iterator& PList<T>::Iterator::operator++()
  {
    ++it;
    ++_index;
    return *this;
  }
  //---------------------------------------------------------------------------
  template <class T> bool PList<T>::Iterator::end() const
  {
    return it == l.end();
  }
  //---------------------------------------------------------------------------
  template <class T> bool PList<T>::Iterator::last() const
  {
    // FIXME: This is probably not the correct way to make the check
    typename std::list<T>::iterator new_it = it;
    ++new_it;
    return new_it == l.end();
  }
  //---------------------------------------------------------------------------
  template <class T> T& PList<T>::Iterator::operator*() const
  {
    return *it;
  }
  //---------------------------------------------------------------------------
  template <class T> T* PList<T>::Iterator::operator->() const
  {
    return &(*it);
  }
  //---------------------------------------------------------------------------
  template <class T> PList<T>::Iterator::operator T*() const
  {
    return &(*it);
  }
  //---------------------------------------------------------------------------
  template <class T> T* PList<T>::Iterator::pointer() const
  {
    return &(*it);
  }
  //---------------------------------------------------------------------------
  template <class T> void PList<T>::Iterator::operator=(const Iterator& it)
  {
    this->it = it;
  }
  //---------------------------------------------------------------------------
  template <class T>  bool PList<T>::Iterator::operator==
  (const Iterator& it) const
  {
    return this->it == it;
  }
  //---------------------------------------------------------------------------
  template <class T>bool PList<T>::Iterator::operator!=
  (const Iterator& it) const
  {
    return this->it != it;
  }
  //---------------------------------------------------------------------------
  
}

#endif
