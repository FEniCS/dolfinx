// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __LIST_H
#define __LIST_H

#include <list>
#include <iterator>
#include <dolfin/dolfin_log.h>

namespace dolfin {

  /// A List is used to store variable-sized sets of data and is implemented
  /// using a double linked list (std::list).

  template <class T> class List {
  public:

    /// Constructor
    List();
    
    /// Destructor
    ~List();
    
    /// Clear list
    void clear();

    /// Add element to the list
    void add(const T& element);
    
    /// Return size of list
    int size() const;

    /// Check if list is empty
    bool empty() const;
    
    /// Iterator for the List class. Should be used as follows:
    ///
    /// for (List<T>::Iterator it(list); !it.end(); ++it) {
    ///     it->...();
    /// }
    
    class Iterator {
    public:
      
      /// Create iterator positioned at the beginning of the list
      Iterator(List<T>& list);

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
  // Implementation of List
  //---------------------------------------------------------------------------
  template <class T> List<T>::List()
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <class T> List<T>::~List()
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <class T> void List<T>::clear()
  {
    l.clear();    
  }
  //---------------------------------------------------------------------------
  template <class T> void List<T>::add(const T& element)
  {
    l.push_back(element);
  }
  //---------------------------------------------------------------------------
  template <class T> int List<T>::size() const
  {
    // Note the following from the stl manual: You should not assume
    // that this function is constant time. It is permitted to be
    // O(N), where N is the number of elements in the list.
    
    return l.size();
  }
  //---------------------------------------------------------------------------
  template <class T> bool List<T>::empty() const
  {
    return l.empty();
  }
  //---------------------------------------------------------------------------
  // Implementation of List::Iterator
  //---------------------------------------------------------------------------
  template <class T> List<T>::Iterator::Iterator (List<T>& list) : l(list.l)
  {
    it = l.begin();
    _index = 0;
  }
  //---------------------------------------------------------------------------
  template <class T> List<T>::Iterator::Iterator(const Iterator& it) : l(it.l)
  {
    this->it = it.it;
    this->_index = it._index;
  }
  //---------------------------------------------------------------------------
  template <class T> int List<T>::Iterator::index() const
  {
    return _index;
  }
  //---------------------------------------------------------------------------
  template <class T> typename List<T>::Iterator& List<T>::Iterator::operator++()
  {
    ++it;
    ++_index;
    return *this;
  }
  //---------------------------------------------------------------------------
  template <class T> bool List<T>::Iterator::end() const
  {
    return it == l.end();
  }
  //---------------------------------------------------------------------------
  template <class T> bool List<T>::Iterator::last() const
  {
    // FIXME: This is probably not the correct way to make the check
    typename std::list<T>::iterator new_it = it;
    ++new_it;
    return new_it == l.end();
  }
  //---------------------------------------------------------------------------
  template <class T> T& List<T>::Iterator::operator*() const
  {
    return *it;
  }
  //---------------------------------------------------------------------------
  template <class T> T* List<T>::Iterator::operator->() const
  {
    return &(*it);
  }
  //---------------------------------------------------------------------------
  template <class T> List<T>::Iterator::operator T*() const
  {
    return &(*it);
  }
  //---------------------------------------------------------------------------
  template <class T> T* List<T>::Iterator::pointer() const
  {
    return &(*it);
  }
  //---------------------------------------------------------------------------
  template <class T> void List<T>::Iterator::operator=(const Iterator& it)
  {
    this->it = it;
  }
  //---------------------------------------------------------------------------
  template <class T>  bool List<T>::Iterator::operator==
  (const Iterator& it) const
  {
    return this->it == it;
  }
  //---------------------------------------------------------------------------
  template <class T>bool List<T>::Iterator::operator!=
  (const Iterator& it) const
  {
    return this->it != it;
  }
  //---------------------------------------------------------------------------
  
}

#endif
