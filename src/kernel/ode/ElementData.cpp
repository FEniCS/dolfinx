// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_settings.h>
#include <dolfin/ElementBlock.h>
#include <dolfin/ElementData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ElementData::ElementData(int N) : N(N), current(0), t0(0), t1(0), empty(true)
{
  // Get size of cache
  cache_size = dolfin_get("element cache size");
}
//-----------------------------------------------------------------------------
ElementData::~ElementData()
{
  // Delete all blocks
  for (BlockIterator block = blocks.begin(); block != blocks.end(); ++block)
  {
    if ( *block )
      delete *block;
    *block = 0;
  }
}
//-----------------------------------------------------------------------------
Element* ElementData::createElement(Element::Type type, 
				    unsigned int q, unsigned int index,
				    real t0, real t1)
{
  // Create a new block if current block is null
  if ( !current )
    createBlock();
  
  // Update time interval
  update(t0, t1);

  // Create a new element in the current block
  return current->createElement(type, q, index, t0, t1);
}
//-----------------------------------------------------------------------------
Element* ElementData::element(unsigned int i, real t)
{
  // Special case: t = t0
  if ( t == t0 )
    return first(i);
  
  // Find a block which *could* contain the element
  ElementBlock* block = findBlock(t);
  if ( block )
    return block->element(i,t);
  
  // Return null otherwise
  return 0;
}
//-----------------------------------------------------------------------------
Element* ElementData::first(unsigned int i)
{
  // Find first block
  ElementBlock* block = findFirst();
  
  // No blocks
  if ( !block )
    return 0;
  
  // Return first element in the first block
  return block->first(i);
}
//-----------------------------------------------------------------------------
Element* ElementData::last(unsigned int i)
{
  // Find last block
  ElementBlock* block = findLast();

  // No blocks
  if ( !block )
    return 0;
  
  // Return last element in the last block
  return block->last(i);
}
//-----------------------------------------------------------------------------
void ElementData::save()
{
  dolfin_assert(current);
  
  // Save current block
  tmpfile.write(*current);
}
//-----------------------------------------------------------------------------
void ElementData::shift()
{
  dolfin_assert(current);
  
  // Set current to null so that a new block will be created next time
  current = 0;
}
//-----------------------------------------------------------------------------
void ElementData::dropLast()
{
  // Pick the last block
  ElementBlock* last = blocks.back();

  // The last block should be the current block
  dolfin_assert(last);
  dolfin_assert(current == last);
  
  // Update interval
  dolfin_assert(t1 == last->endtime());
  t1 = last->starttime();

  // Delete last block and remove it from the list
  delete last;
  blocks.pop_back();

  // Set current to null so that a new block will be created next time
  current = 0;
}
//-----------------------------------------------------------------------------
unsigned int ElementData::size() const
{
  return N;
}
//-----------------------------------------------------------------------------
bool ElementData::within(real t) const
{
  return (t0 < t) && (t <= t1);
}
//-----------------------------------------------------------------------------
void ElementData::createBlock()
{
  // Current should be null when we create a new block
  dolfin_assert(current == 0);

  // Remove old block if necessary
  if ( memoryFull() )
    dropBlock(t0,t1);
  
  // Create new block
  current = new ElementBlock(N);
  blocks.push_back(current);
}
//-----------------------------------------------------------------------------
ElementBlock* ElementData::findBlock(real t)
{
  // First check current block
  if ( current )
  {
    if ( current->within(t) )
    {
      return current;
    }
  }
  // Return null if time is not within range of data
  if ( !within(t) )
    return 0;

  // Next, check all blocks
  for (BlockIterator block = blocks.begin(); block != blocks.end(); ++block)
    if ( (*block)->within(t) )
      return *block;

  // Create a new block
  createBlock();

  // Read the block from the file
  tmpfile.read(*current, t);

  return current;
}
//-----------------------------------------------------------------------------
ElementBlock* ElementData::findFirst()
{
  // First check current block
  if ( current )
    if ( current->starttime() == t0 )
      return current;

  // Next, check all blocks
  for (BlockIterator block = blocks.begin(); block != blocks.end(); ++block)
    if ( (*block)->starttime() == t0 )
      return *block;

  // Check if there are any blocks stored on file
  if ( tmpfile.empty() )
    return 0;

  // Create a new block
  createBlock();

  // Read the first block from the file
  tmpfile.readFirst(*current);

  return current;
}
//-----------------------------------------------------------------------------
ElementBlock* ElementData::findLast()
{
  // First check current block
  if ( current )
    if ( current->endtime() == t1 )
      return current;

  // Next, check all blocks
  for (BlockIterator block = blocks.begin(); block != blocks.end(); ++block)
    if ( (*block)->endtime() == t1 )
      return *block;

  // Check if there are any blocks stored on file
  if ( tmpfile.empty() )
    return 0;

  // Create a new block
  createBlock();

  // Read the last block from the file
  tmpfile.readLast(*current);

  return current;
}
//-----------------------------------------------------------------------------
void ElementData::update(real t0, real t1)
{
  // Save interval if this is the first element
  if ( empty )
  {
    this->t0 = t0;
    this->t1 = t1;
    empty = false;
    return;
  }

  // Otherwise, check the boundaries
  if ( t0 < this->t0 )
    this->t0 = t0;
  if ( t1 > this->t1 )
    this->t1 = t1;
}
//-----------------------------------------------------------------------------
bool ElementData::memoryFull()
{
  // Memory not full if there are no blocks
  if ( blocks.empty() )
    return false;

  // Count number of bytes for all blocks
  unsigned int bytecount = 0;
  for (BlockIterator block = blocks.begin(); block != blocks.end(); ++block)
    bytecount += (*block)->bytes();

  // Estimate data size including a new block (using last block as estimate)
  bytecount += blocks.back()->bytes();

  // Get size in MB
  bytecount /= DOLFIN_MEGABYTE;

  return bytecount > cache_size;
}
//-----------------------------------------------------------------------------
void ElementData::dropBlock(real t0, real t1)
{
  // Find which block is the one furthest from [t0,t1]
  BlockIterator last = blocks.end();
  real dist = 0.0;
  
  for (BlockIterator block = blocks.begin(); block != blocks.end(); ++block)
  {
    real d = (*block)->dist(t0,t1);
    if ( d >= dist )
    {
      dist = d;
      last = block;
    }
  }
  
  // Delete the last block
  dolfin_assert(last != blocks.end());
  if ( *last == current )
    current = 0;
  delete *last;
  blocks.erase(last);
}
//-----------------------------------------------------------------------------
