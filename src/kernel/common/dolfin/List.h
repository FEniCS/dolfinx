// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __LIST_H
#define __LIST_H

// List implements a block-linked list.
//
// Objects can be dynamically added to (or removed from) the list.
// New blocks of size BLOCK_SIZE are allocated when needed.

#include <iostream>
#include <dolfin/Display.h>

#define BLOCK_SIZE 1024

template <class T> class List{
public:

  /// Constructor
  List(){
	 
	 _first_block = 0;
	 _current_block = _first_block;
	 
	 _blocks = 0;
	 _size = 0;
	 _empty = 0;

  }

  /// Destructor
  ~List(){

	 // Delete all blocks
	 if ( _first_block ){
		Block *b = _first_block;
		while ( true ){
		  
		  if ( b->prev )
			 delete b->prev;
		  
		  if ( !b->next )
			 break;
		  
		  b = b->next;
		}
		delete b;
	 }
	 
  }
  
  /// Creates a new object in the list and returns a unique id
  T* create(int *id){

	 // Create a new block if there are no blocks
	 if ( !_first_block ){
		_first_block = new Block(0);
		_current_block = _first_block;
		_blocks = 1;
	 }
	 
	 // Check for empty position
	 if ( _empty > 0 ){
		for (Block *b = _first_block;; b = b->next){
		  if ( !(b->full()) ){
			 _size += 1;
			 _empty -= 1;			 
			 return b->create(id);
		  }
		  if ( !b->next )
			 break;
		}
		display->InternalError("List::create()","Cannot find an empty position");
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

  // Creates a new object in the list
  T* create(){
	 
	 int dummy;
	 T* t = create(&dummy);
	 return t;
  }
  
  /// Adds an object to the list
  T* add(T x){
	 T *new_x = create();
	 *new_x = x;
  }

  /// Returns a pointer to the object with the given id
  T* pointer(int id){

	 // Check current block
	 if ( id >= _current_block->number*BLOCK_SIZE && id < (_current_block->number+1)*BLOCK_SIZE )
		return _current_block->pointer(id);

	 // Check all blocks
	 for (Block *b = _first_block;; b = b->next){

		if ( id >= b->number*BLOCK_SIZE && id < (b->number+1)*BLOCK_SIZE )
		  return b->pointer(id);

		if ( !b->next )
		  break;

	 }

	 // No object with given id
	 return 0;

  }
  
  /// True if the list is empty
  bool empty(){
	 return _size == 0;
  }

  /// Returns the size of the list
  bool size(){
	 return _size;
  }

  /// Output
  friend std::ostream &operator<<(std::ostream &os, const List<T> &list)
  {
	 os << "size = " << list._size << " ";
	 os << "(" << list._blocks*BLOCK_SIZE << " objects allocated in " << list._blocks << " blocks)";

	 return os;
  }
  
  class Iterator;
  
  //---------------------------------------------------------------------------

  class Block{
  public:
	 
	 friend class List<T>;
	 friend class Iterator;
	 
	 Block(Block *prev){

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
		if ( !(data = new T[BLOCK_SIZE]) )
		  display->Error("Out of memory");
		if ( !(empty = new bool[BLOCK_SIZE]) )
		  display->Error("Out of memory");
		for (int i=0;i<BLOCK_SIZE;i++)
		  empty[i] = true;
		pos = 0;
		used = 0;
		
	 }
	 
	 ~Block(){
		delete [] data;
		delete [] empty;
	 }

	 T* pointer(int id){
		id -= number*BLOCK_SIZE;

		if ( id < 0 || id >= BLOCK_SIZE )
		  return 0;

		return data + id;
	 }
	 
	 bool full(){
		return used == BLOCK_SIZE;
	 }
	 
	 int first_pos(){
		for (int i=0;i<BLOCK_SIZE;i++)
		  if ( !empty[i] )
			 return i;
		
		return -1;
	 }
	 
	 int next_pos(int start){
		for (int i=start;i<BLOCK_SIZE;i++)
		  if ( !empty[i] )
			 return i;
		
		return -1;
	 }
	 
	 T* create(int *id){

		// Check if the list is full
		if ( used == BLOCK_SIZE )
		  display->InternalError("Block::create()","Block is full");
		
		// Check if there is an empty position
		if ( used != pos ){
		  for (int i=0;i<BLOCK_SIZE;i++)
			 if ( empty[i] ){
				empty[i] = false;
				used += 1;
				*id = number*BLOCK_SIZE + i;
				return data + i;
			 }
		  display->InternalError("Block::create()","Unable to find an empty position");
		}
		
		// Use next available position
		empty[pos] = false;
		*id = number*BLOCK_SIZE + pos;
		pos += 1;
		used += 1;
		return data + pos - 1;
	 }
	 
  protected:
	 
	 // Pointer to previous and next blocks
	 Block *prev;
	 Block *next;
	 
	 // Data
	 T *data;
	 bool *empty;
	 
	 // Next available position and size of list
	 int pos;
	 int used;

	 // Number of this block
	 int number;
	 
  };
  
  //---------------------------------------------------------------------------
  
  class Iterator{
  public:
	 
	 friend class List<T>;
	 
	 // Create iterator positioned at the end of the list
	 Iterator(){
		list = 0;
		block = 0;
		pos = 0;
		at_end = true;
	 }

	 // Create iterator positioned at the beginning of the list
	 Iterator(List<T> *list){
		this->list = list;
		list->iterator_start(this);
	 }

	 /// Prefix increment
	 List<T>::Iterator& operator++()
	 {
		if ( list )
		  list->iterator_step(this);
		return *this;
	 }

	 /// Returns current object
	 T& operator*() const
	 {
		return block->data[pos];
	 }

	 /// Pointer mechanism
	 T* operator->() const
	 {
		return block->data+pos;
	 }

	 /// Returns pointer to current object
	 T* pointer() const
	 {
		return block->data+pos;
	 }

	 /// Equality operator
	 bool operator==(const Iterator& it) const
	 {
		if ( at_end && it.at_end )
		  return true;
		
		return list == it.list && block == it.block && pos == it.pos;
	 }

	 /// Inequality operator
	 bool operator!=(const Iterator& it) const
	 {
		return !( *this == it );
	 }

	 /// Output
	 friend std::ostream &operator<<(std::ostream &os, const Iterator &it)
	 {
		return os << "block = " << it.block << " pos = " << it.pos;
	 }
	 
  protected:
	 List<T> *list;
	 Block *block;
	 int pos;
	 bool at_end;
  };
  
  //---------------------------------------------------------------------------

  /// Returns an iterator positioned at the beginning of the list
  Iterator begin()
  {
	 return Iterator(this);
  }

  /// Returns an iteratator positioned at the end of the list
  Iterator end()
  {
	 return Iterator();
  }
  
  friend class Block;
  friend class Iterator;
  
protected:
  
  // Place iterator at the beginning of the list
  void iterator_start(Iterator *it){
	 
	 // Check if the list is empty
	 if ( _size == 0 ){
		it->at_end = true;
		return;
	 }
	 
	 // List is not empty
	 it->at_end = false;
	 
	 // Place iterator at the beginning of the list
	 for (Block *b = _first_block;; b = b->next){

		int pos;
		
		if ( (pos = b->first_pos()) != -1 ){
		  it->block = b;
		  it->pos = pos;
		  return;
		}

		if ( !b->next )
		  break;

	 }

	 // Something strange happened
	 display->InternalError("List::iterator_start()","Unable to find first object in list");
  }
  
  // Step iterator to the next position in the list
  void iterator_step(Iterator *it){
	 
	 // Check if the list is empty
	 if ( _size == 0 ){
		it->at_end = true;
		return;
	 }
	 
	 // Step to next position
	 int start = it->pos + 1;
	 int pos = 0;
	 for (Block *b = it->block;; b = b->next){
		

		
		if ( (pos = b->next_pos(start)) != -1 ){
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

#endif
