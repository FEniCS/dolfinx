// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __LIST_H
#define __LIST_H

// List implements a block-linked list.
//
// Elements can be dynamically added to (or removed from) the list.
// New blocks of size BLOCK_SIZE are allocated when needed.

#include <dolfin/dolfin_log.h>

namespace dolfin {

  template <class T> class List {
  public:

    // Forward declaration of nested classes
    class Block;
    class Iterator;
	 
    /// Constructor
    List();
    
    /// Destructor
    ~List();

    /// Create a new element in the list
    T* create();

    /// Create a new element in the list
    T* create(int* id);
	 
    /// Add an element to the list
    T* add(T x);
	 
    /// Return a pointer to the element with the given id
    T* pointer(int id);
	 
    /// Check if the list is empty
    bool empty() const;
	 
    /// Return the size of the list
    int size() const;
	 
    /// Output
    friend LogStream& operator<< <> (LogStream& stream, const List<T>& list);

    /// Return an iterator positioned at the beginning of the list
    Iterator begin();

    /// Return an iterator positioned at the end of the list
    Iterator end();

    /// Return an iterator positioned at the element with given id
    Iterator iterator(int id);

    /// Friends
    friend class Block;
    friend class Iterator;
 
    /// Iterator for the List class. Should be used as follows:
    ///
    ///     for (List<T>::Iterator it(list); !it.end(); ++it) {
    ///         it->...();
    ///      }

    class Iterator {
    public:
      		
      /// Create iterator positioned at the end of the list
      Iterator();
      
      /// Copy constructor
      Iterator(const Iterator& it);

      /// Create iterator positioned at the beginning of the list
      Iterator(List<T>& list);
		
      /// Create iterator positioned at the element with given id
      Iterator(List<T>& list, int id);

      /// Current position in the list
      int index() const;
		
      /// Prefix increment
      List<T>::Iterator& operator++();
		
      /// Check if iterator has reached end of list
      bool end() const;
		
      /// Check if iterator is at the last position
      bool last() const;
		
      /// Return current element
      T& operator*() const;

      /// Pointer mechanism
      T* operator->() const;
		
      /// Returns pointer to current element
      T* pointer() const;
		
      /// Assignment
      void operator=(const Iterator& it);

      /// Equality operator
      bool operator==(const Iterator& it) const;
		
      /// Inequality operator
      bool operator!=(const Iterator& it) const;

      /// Friends
      friend class List<T>;
	
    private:

      List<T>* list;
      Block* block;
      int pos;
      int _index;
      bool at_end;

    };
	 	 
  private:
    
    class Block {
    public:
      
      friend class List<T>;
      friend class Iterator;
		
      Block(Block* prev);
      ~Block();
		
      T* pointer(int id);
      int position(int id);
      bool full();
      int first_pos();
      int next_pos(int start);
      T* create(int* id);
		
    private:
		
      // Pointer to previous and next blocks
      Block* prev;
      Block* next;
		
      // Data
      T* data;
      bool* empty;
      
      // Next available position and size of list
      int pos;
      int used;
		
      // Number of this block
      int number;
      
    };
	 
    // Place iterator at the beginning of the list
    void iterator_start(Iterator* it);

    // Place iterator at the element with given id
    void iterator_id(Iterator* it, int id);
	 
    // Step iterator to the next position in the list
    void iterator_step(Iterator* it);

    // Pointers to first and current blocks
    Block *_first_block;
    Block *_current_block;
	 
    // Number of blocks and size of list
    int _blocks;
    int _size;
    int _empty;
	 
    // True if there is an empty position somewhere in the list
    bool _empty_position;
	 
  };

  //---------------------------------------------------------------------------
  // Implementation of List
  //---------------------------------------------------------------------------
  template <class T> List<T>::List()
  {
    _first_block = 0;
    _current_block = _first_block;
    
    _blocks = 0;
    _size = 0;
    _empty = 0;  
  }
  //---------------------------------------------------------------------------
  template <class T> List<T>::~List()
  {
    // Delete all blocks
    if ( _first_block ) {
      
      Block *b = _first_block;
      while ( true ) {
	
	if ( b->prev )
	  delete b->prev;
	
	if ( !b->next )
	  break;
	
	b = b->next;
	
      }
      
      delete b;
    }
    
  }
  //---------------------------------------------------------------------------
  template <class T> T* List<T>::create()
  {
    int dummy;
    T* t = create(&dummy);
    return t;
  }
  //---------------------------------------------------------------------------
  template <class T> T* List<T>::create(int* id)
  {
    // Create a new block if there are no blocks
    if ( !_first_block ){
      _first_block = new Block(0);
      _current_block = _first_block;
      _blocks = 1;
    }
    
    // Check for empty position
    if ( _empty > 0 ){
      for (Block *b = _first_block;; b = b->next) {
	if ( !(b->full()) ){
	  _size += 1;
	  _empty -= 1;			 
	  return b->create(id);
	}
	if ( !b->next )
	  break;
      }
      dolfin_error("Unable to find empty position in list.");
    }
    
    // Use next empty position
    if ( !_current_block->full() ){
      _size += 1;
      _empty -= 1;
      return _current_block->create(id);
    }
    
    // Create new block
    _current_block = new Block(_current_block);
    _size += 1;
    _empty -= 1;
    _blocks += 1;
    
    return _current_block->create(id);
  }
  //---------------------------------------------------------------------------
  template <class T> T* List<T>::add(T x)
  {
    T *new_x = create();
    *new_x = x;
  }
  //---------------------------------------------------------------------------
  template <class T> T* List<T>::pointer(int id)
  {
    // Check current block
    if ( id >= _current_block->number*DOLFIN_BLOCK_SIZE && 
	 id < (_current_block->number+1)*DOLFIN_BLOCK_SIZE )
      return _current_block->pointer(id);
    
    // Check all blocks
    for (Block *b = _first_block;; b = b->next){
      
      if ( id >= b->number*DOLFIN_BLOCK_SIZE && 
	   id < (b->number+1)*DOLFIN_BLOCK_SIZE )
	return b->pointer(id);
      
      if ( !b->next )
	break;
      
    }
    
    // No element with given id
    return 0;
  }
  //---------------------------------------------------------------------------	 
  template <class T> bool List<T>::empty() const
  {
    return _size == 0;
  }
  //---------------------------------------------------------------------------	 
  template <class T> int List<T>::size() const
  {
    return _size;
  }
  //---------------------------------------------------------------------------
  template <class T> LogStream& operator<<(LogStream& stream, const List<T> &list)
  {
    stream << "[ Block-linked list of size " << list._size << ". ";
    stream << list._blocks*DOLFIN_BLOCK_SIZE << " elements allocated in " 
	   << list._blocks << " blocks.]";
    
    return stream;
  }
  //---------------------------------------------------------------------------
  template <class T> List<T>::Iterator List<T>::begin()
  {
    return Iterator(*this);
  }
  //---------------------------------------------------------------------------	 
  template <class T> List<T>::Iterator List<T>::end()
  {
    return Iterator();
  }
  //---------------------------------------------------------------------------
  template <class T> List<T>::Iterator List<T>::iterator(int id)
  {
    return Iterator(*this, id);
  }
  //---------------------------------------------------------------------------
  template <class T> void List<T>::iterator_start(Iterator* it)
  {
    // Check if the list is empty
    if ( _size == 0 ) {
      it->at_end = true;
      return;
    }
    
    // List is not empty
    it->at_end = false;
    
    // Place iterator at the beginning of the list
    for (Block *b = _first_block;; b = b->next) {
      
      int pos;
      
      if ( (pos = b->first_pos()) != -1 ) {
	it->block = b;
	it->pos = pos;
	return;
      }
      
      if ( !b->next )
	break;
      
    }
    
    // Something strange happened
    dolfin_error("Unable to find first element in list.");
  }
  //---------------------------------------------------------------------------
  template <class T> void List<T>::iterator_id(Iterator* it, int id)
  {
    it->at_end = false;

    // Check current block
    if ( id >= _current_block->number*DOLFIN_BLOCK_SIZE && 
	 id < (_current_block->number+1)*DOLFIN_BLOCK_SIZE ) {
      it->block = _current_block;
      it->pos = _current_block->position(id);
      return;
    }
         
    // Check all blocks
    for (Block *b = _first_block;; b = b->next) {
      
      if ( id >= b->number*DOLFIN_BLOCK_SIZE && 
	   id < (b->number+1)*DOLFIN_BLOCK_SIZE ) {
	it->block = _current_block;
	it->pos = _current_block->position(id);
	return;
      }
      
      if ( !b->next )
	break;
      
    }
    
    // No element with given id
    dolfin_error("Unable to create iterator for given id.");
  }
  //---------------------------------------------------------------------------
  template <class T> void List<T>::iterator_step(Iterator* it)
  {
    // Check if the list is empty
    if ( _size == 0 ) {
      it->at_end = true;
      return;
    }
    
    // Step to next position
    int start = it->pos + 1;
    int pos = 0;
    for (Block *b = it->block;; b = b->next) {
      
      if ( (pos = b->next_pos(start)) != -1 ) {
	it->block = b;
	it->pos = pos;
	return;
      }
      
      if ( !b->next )
	break;
      
      start = 0;
    }
		
    // We have reached the end of the list
    it->at_end = true;
  }
  //---------------------------------------------------------------------------
  // Implementation of List<T>::Iterator
  //---------------------------------------------------------------------------
  template <class T> List<T>::Iterator::Iterator()
  {
    list = 0;
    
    block = 0;
    _index = -1;
    pos = 0;
    
    at_end = true;
  }
  //---------------------------------------------------------------------------
  template <class T> List<T>::Iterator::Iterator(const Iterator& it)
  {
    list   = it.list;
    block  = it.block;
    pos    = it.pos;
    _index = it._index;
    at_end = it.at_end;    
  }
  //---------------------------------------------------------------------------
  template <class T> List<T>::Iterator::Iterator(List<T>& list)
  {
    this->list = &list;
    list.iterator_start(this);
    _index = 0;
  }
  //---------------------------------------------------------------------------
  template <class T> List<T>::Iterator::Iterator(List<T>& list, int id)
  {
    this->list = &list;
    list.iterator_id(this, id);
    _index = id;
  }
  //---------------------------------------------------------------------------
  template <class T> int List<T>::Iterator::index() const
  {
    return _index;
  }
  //---------------------------------------------------------------------------		
  template <class T> List<T>::Iterator& List<T>::Iterator::operator++()
  {
    if ( list ) {
      list->iterator_step(this);
      _index++;
    }
    return *this;
  }
  //---------------------------------------------------------------------------		
  template <class T> bool List<T>::Iterator::end() const
  {
    return at_end;
  }
  //---------------------------------------------------------------------------		
  template <class T> bool List<T>::Iterator::last() const
  {
    return _index == (list->size() - 1);
  }
  //---------------------------------------------------------------------------		
  template <class T> T& List<T>::Iterator::operator*() const
  {
    return block->data[pos];
  }
  //---------------------------------------------------------------------------		
  template <class T> T* List<T>::Iterator::operator->() const
  {
    return block->data+pos;
  }
  //---------------------------------------------------------------------------		
  template <class T>  T* List<T>::Iterator::pointer() const
  {
    return block->data+pos;
  }
  //---------------------------------------------------------------------------		
  template <class T> void List<T>::Iterator::operator=(const Iterator& it)
  {
    list   = it.list;
    block  = it.block;
    pos    = it.pos;
    _index = it._index;
    at_end = it.at_end;
  }
  //---------------------------------------------------------------------------		
  template <class T> bool List<T>::Iterator::operator==
  (const Iterator& it) const
  {
    if ( at_end && it.at_end )
      return true;
    
    return list == it.list && block == it.block && pos == it.pos;
  }
  //---------------------------------------------------------------------------		
  template <class T> bool List<T>::Iterator::operator!=
  (const Iterator& it) const
  {
    return !( *this == it );
  }
  //---------------------------------------------------------------------------
  // Implementation of List<T>::Block
  //---------------------------------------------------------------------------
  template <class T> List<T>::Block::Block(Block* prev)
  {
    // Set block number
    if ( prev )
      number = prev->number + 1;
    else
      number = 0;
    
    // Link blocks
    if ( prev )
      prev->next = this;
    this->prev = prev;
    next = 0;
    
    // Allocate memory for this block
    if ( !(data = new T[DOLFIN_BLOCK_SIZE]) )
      dolfin_error("Out of memory.");
    if ( !(empty = new bool[DOLFIN_BLOCK_SIZE]) )
      dolfin_error("Out of memory.");
    for (int i=0;i<DOLFIN_BLOCK_SIZE;i++)
      empty[i] = true;
    pos = 0;
    used = 0;
    
  }
  //---------------------------------------------------------------------------		
  template <class T> List<T>::Block::~Block()
  {
    delete [] data;
    delete [] empty;
  }
  //---------------------------------------------------------------------------	
  template <class T> T* List<T>::Block::pointer(int id)
  {
    id -= number*DOLFIN_BLOCK_SIZE;
    
    if ( id < 0 || id >= DOLFIN_BLOCK_SIZE )
      return 0;
    
    return data + id;
  }
  //---------------------------------------------------------------------------	
  template <class T> int List<T>::Block::position(int id)
  {
    return id - number*DOLFIN_BLOCK_SIZE;
  }
  //---------------------------------------------------------------------------  
  template <class T> bool List<T>::Block::full()
  {
    return used == DOLFIN_BLOCK_SIZE;
  }
  //---------------------------------------------------------------------------
  template <class T> int List<T>::Block::first_pos()
  {
    for (int i =0 ; i < DOLFIN_BLOCK_SIZE; i++)
      if ( !empty[i] )
	return i;
    
    return -1;
  }
  //---------------------------------------------------------------------------  
  template <class T> int List<T>::Block::next_pos(int start)
  {
    for (int i = start; i < DOLFIN_BLOCK_SIZE; i++)
      if ( !empty[i] )
	return i;
    
    return -1;
  }
  //---------------------------------------------------------------------------
  template <class T> T* List<T>::Block::create(int* id)
  {
    // Check if the list is full
    if ( used == DOLFIN_BLOCK_SIZE )
      dolfin_error("Block is full.");
    
    // Check if there is an empty position
    if ( used != pos ){
      for (int i = 0; i < DOLFIN_BLOCK_SIZE; i++)
	if ( empty[i] ){
	  empty[i] = false;
	  used += 1;
	  *id = number*DOLFIN_BLOCK_SIZE + i;
	  return data + i;
	}
      dolfin_error("Unable to find an empty position.");
    }
    
    // Use next available position
    empty[pos] = false;
    *id = number*DOLFIN_BLOCK_SIZE + pos;
    pos += 1;
    used += 1;
    return data + pos - 1;
  }
  //---------------------------------------------------------------------------
  
}
  
#endif
