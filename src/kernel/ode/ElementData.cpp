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
  cachesize = dolfin_get("element cache size");
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
Element* ElementData::createElement(const Element::Type type, real t0, real t1,
				    int q, int index)
{
  // Create a new block if current block is null
  if ( !current )
  {
    // Remove old block if necessary
    if ( memfull() )
      droplast(t0,t1);
    
    // Create new block
    current = new ElementBlock(N);
    blocks.push_back(current);
    cout << "Number of blocks: " << blocks.size() << endl;
  }

  update(t0, t1);
  return current->createElement(type, t0, t1, q, index);
}
//-----------------------------------------------------------------------------
Element* ElementData::element(unsigned int i, real t)
{
  // Find a block which *could* contain the element
  ElementBlock* block = findpos(t);
  if ( block )
    return block->element(i,t);
  
  // Return null otherwise
  return 0;
}
//-----------------------------------------------------------------------------
Element* ElementData::last(unsigned int i)
{
  return current->last(i);
}
//-----------------------------------------------------------------------------
void ElementData::shift()
{
  dolfin_assert(current);
  
  // Save current block
  current->save();

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
ElementBlock* ElementData::findpos(real t)
{
  // First check current block
  if ( current )
    if ( current->within(t) )
      return current;

  // Return null if time is not within range of data
  if ( !within(t) )
    return 0;
  
  cout << "Not in current block" << endl;

  // Next, check all blocks
  for (BlockIterator block = blocks.begin(); block != blocks.end(); ++block)
    if ( (*block)->within(t) )
      return *block;

  // Didn't find a block
  return 0;
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
bool ElementData::memfull()
{
  // Memory not full if there are no blocks
  if ( blocks.empty() )
    return false;

  // Count number of bytes for all blocks
  unsigned int bytecount = 0;
  for (BlockIterator block = blocks.begin(); block != blocks.end(); ++block)
    bytecount += (*block)->bytes();

  cout << "Number of bytes: " << bytecount << endl;

  // Estimate data size including a new block (using last block as estimate)
  bytecount += blocks.back()->bytes();

  return bytecount > cachesize;
}
//-----------------------------------------------------------------------------
void ElementData::droplast(real t0, real t1)
{
  // Find which block is the one furthest from [t0,t1]
  BlockIterator last = blocks.end();
  real dist = 0;
  
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
  blocks.erase(last);
}
//-----------------------------------------------------------------------------
