// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/Element.h>
#include <dolfin/Component.h>
#include <dolfin/ElementBlock.h>
#include <dolfin/NewArray.h>
#include <dolfin/ElementTmpFile.h>

using namespace dolfin;

// The temporary file will be created in the directory specified by
// P_tmpdir in <stdio.h> (which is often /tmp) and will be
// automatically deleted when the program exits. The file will not be
// visible during execution (since tmpfile() directly calls unlink()).
//
// Data is stored sequentially and binary block by block, where each
// block is stored in the following way
//
//   size of block (in bytes)                      unsigned int
//   start time of block                           real
//   end time of block                             real
//   number of components                          unsigned int
//
//   number of elements for component 0            unsigned int
//
//     type for element 0 of component 0           unsigned int
//     order for element 0 of component 0          unsigned int
//     start time for element 0 of component 0     real
//     end time for element 0 of component 0       real
//       value 0 of element 0 of component 0       real
//       value 1 of element 0 of component 0       real
//       ...                                  
//       value q of element 0 of component 0       real
//
//     type for element 1 of component 0           unsigned int
//     order for element 1 of component 0          unsigned int
//     start time for element 1 of component 0     real
//     end time for element 1 of component 0       real
//       value 0 of element 1 of component 0       real
//       value 1 of element 1 of component 0       real
//       ...                                  
//       value q of element 1 of component 0       real
//     ...
//     type for last element of component 0        unsigned int
//     order for last element of component 0       unsigned int
//     start time for last element of component 0  real
//     end time for last element of component 0    real
//       value 0 of last element of component 0    real
//       value 1 of last element of component 0    real
//       ...     
//       value q of last element of component 0    real
//
//   number of elements for component 1            unsigned int
//     ...
//   number of elements for component N-1          unsigned int
//     ...
//
//   size of block (in bytes)                      unsigned int
//
// Note that the size of the block is stored both at the beginning and
// at the end of each block, to allow search forwards and backwards.

//-----------------------------------------------------------------------------
ElementTmpFile::ElementTmpFile() : _empty(true)
{
  // Open file
  fp = tmpfile();
}
//-----------------------------------------------------------------------------
ElementTmpFile::~ElementTmpFile()
{
  // Close file
  fclose(fp);
}
//-----------------------------------------------------------------------------
void ElementTmpFile::write(const ElementBlock& block)
{
  // Write size of block (number of bytes)
  unsigned int size = bytes(block);
  fwrite(&size, sizeof(unsigned int), 1, fp);

  // Write time span of block
  real t0 = block.t0;
  real t1 = block.t1;
  fwrite(&t0, sizeof(real), 1, fp);
  fwrite(&t1, sizeof(real), 1, fp);

  // Write the number of components
  unsigned int N = block.size();
  fwrite(&N, sizeof(unsigned int), 1, fp);

  // Write data for each component
  for (unsigned int i = 0; i < block.size(); i++)
  {
    // Get list of elements for current component
    const NewArray<Element*>& elements = block.components[i].elements;

    // Write number of elements
    unsigned int size = elements.size();
    fwrite(&size, sizeof(unsigned int), 1, fp);

    // Write data for each element
    typedef NewArray<Element*>::const_iterator ElementIterator;
    for (ElementIterator e = elements.begin(); e != elements.end(); ++e)
    {
      // Write type of element
      unsigned int type = 0;
      if ( (*e)->type() == Element::cg )
	type = 0;
      else
	type = 1;
      fwrite(&type, sizeof(unsigned int), 1, fp);

      // Write order of element
      unsigned int order = (*e)->order();
      fwrite(&order, sizeof(unsigned int), 1, fp);

      // Write time span of element
      real t0 = (*e)->starttime();
      real t1 = (*e)->endtime();
      fwrite(&t0, sizeof(real), 1, fp);
      fwrite(&t1, sizeof(real), 1, fp);

      // Write initial value for dG element
      if ( (*e)->type() == Element::dg )
      {
	real initval = (*e)->initval();
	fwrite(&initval, sizeof(real), 1, fp);
      }
      
      // Write element values
      for (unsigned int i = 0; i <= order; i++)
      {
	real value = (*e)->value(i);
	fwrite(&value, sizeof(real), 1, fp);
      }
    }
  }

  // Write size of block (number of bytes)
  fwrite(&size, sizeof(unsigned int), 1, fp);

  // Remember that the file is not empty
  _empty = false;
}
//-----------------------------------------------------------------------------
void ElementTmpFile::read(ElementBlock& block, real t)
{
  // Check if the file is empty
  if ( _empty )
    dolfin_error("Unable to read element data. File is empty.");

  // Search forward
  if ( searchForward(t) )
  {
    readBlock(block);
    return;
  }

  // Search backward
  if ( searchBackward(t) )
  {
    readBlock(block);
    return;
  }

  dolfin_error("Unable to find element data in file.");
}
//-----------------------------------------------------------------------------
void ElementTmpFile::readFirst(ElementBlock& block)
{
  // Step to the beginning of the file
  rewind(fp);

  // Read block data from current position
  readBlock(block);
}
//-----------------------------------------------------------------------------
void ElementTmpFile::readLast(ElementBlock& block)
{
  // Step to the end of the file
  if ( fseek(fp, 0L, SEEK_END) != 0 )
    dolfin_error("Unable to read element data.");

  // Read tail of block
  unsigned int size = 0;
  if ( !readTail(size) )
    return;
  
  // Skip backward
  if ( fseek(fp, -size, SEEK_CUR) != 0 )
    return;

  // Step to the beginning of the block
  if ( fseek(fp, -size, SEEK_CUR) != 0 )
    dolfin_error("Unable to read element data.");
  
  // Read block data from current position
  readBlock(block);
}
//-----------------------------------------------------------------------------
bool ElementTmpFile::empty() const
{
  return _empty;
}
//-----------------------------------------------------------------------------
bool ElementTmpFile::searchForward(real t)
{
  unsigned int size = 0;
  real t0 = 0.0;
  real t1 = 0.0;

  // Search forward and try to find t0 < t <= t1
  while (true)
  {
    // Read head of block
    if ( !readHead(size, t0, t1) )
      return false;

    // Check if we found the block
    if ( t0 < t && t <= t1 )
      return true;
    
    // Check that we have not missed the block
    if ( t <= t0 )
      return false;

    // Skip forward
    if ( fseek(fp, size, SEEK_CUR) != 0 )
      return false;
  }

  return false;
}
//-----------------------------------------------------------------------------
bool ElementTmpFile::searchBackward(real t)
{
  unsigned int size = 0;
  real t0 = 0.0;
  real t1 = 0.0;

  // Search backward and try to find t0 < t <= t1
  while (true)
  {
    // Read tail of block
    if ( !readTail(size) )
      return false;

    // Skip backward
    if ( fseek(fp, -size, SEEK_CUR) != 0 )
      return false;

    // Read head of block
    if ( !readHead(size, t0, t1) )
      return false;

    // Check if we found the block
    if ( t0 < t && t <= t1 )
      return true;
    
    // Check that we have not missed the block
    if ( t > t1 )
      return false;
  }

  return false;
}
//-----------------------------------------------------------------------------
bool ElementTmpFile::readHead(unsigned int& size, real& t0, real &t1)
{
  // Read size
  if ( fread(&size, sizeof(unsigned int), 1, fp) == 0 )
    return false;
  
  // Read t0
  if ( fread(&t0, sizeof(real), 1, fp) == 0 )
    return false;

  // Read t1
  if ( fread(&t1, sizeof(real), 1, fp) == 0 )
    return false;

  // Step back to beginning of block
  long offset = sizeof(unsigned int) + 2*sizeof(real);
  if ( fseek(fp, -offset, SEEK_CUR) != 0 )
    return false;

  return true;
}
//-----------------------------------------------------------------------------
bool ElementTmpFile::readTail(unsigned int& size)
{
  // Step back within the previous block
  if ( fseek(fp, -sizeof(unsigned int), SEEK_CUR)  )
    return false;

  // Read size
  if ( fread(&size, sizeof(unsigned int), 1, fp) == 0 )
    return false;

  return true;
}
//-----------------------------------------------------------------------------
void ElementTmpFile::readBlock(ElementBlock& block)
{
  // Read size of block (number of bytes)
  unsigned int size = 0;
  fread(&size, sizeof(unsigned int), 1, fp);
  
  // Check that the file is not empty
  if ( feof(fp) )
    dolfin_error("Unable to read element data.");

  // Read time span of block
  real t0 = 0.0;
  real t1 = 0.0;
  fread(&t0, sizeof(real), 1, fp);
  fread(&t1, sizeof(real), 1, fp);

  // Read number of components
  unsigned int N = 0;
  fread(&N, sizeof(unsigned int), 1, fp);

  // Check that we got the correct number of components
  if ( N != block.size() )
    dolfin_error("Incorrect block size.");

  // Read data for each component
  for (unsigned int index = 0; index < N; index++)
  {
    // Read number of elements for component
    unsigned int size = 0;
    fread(&size, sizeof(unsigned int), 1, fp);
    
    // Read data for each element
    typedef NewArray<Element*>::const_iterator ElementIterator;
    for (unsigned int i = 0; i < size; i++)
    {
      // Read type of element
      unsigned int type = 0;
      fread(&type, sizeof(unsigned int), 1, fp);

      // Read order of element
      unsigned int order = 0;
      fread(&order, sizeof(unsigned int), 1, fp);

      // Read time span of element
      real t0 = 0.0;
      real t1 = 0.0;
      fread(&t0, sizeof(real), 1, fp);
      fread(&t1, sizeof(real), 1, fp);

      // Create the element (assume block is empty at start)
      Element* element = 0;
      switch (type) {
      case 0:
	element = block.createElement(Element::cg, order, index, t0, t1);
	break;
      case 1:
	element = block.createElement(Element::dg, order, index, t0, t1);
	break;
      default:
	dolfin_error("Unknown element type.");
      }
      dolfin_assert(element);

      // Read initial value for dG element
      if ( type == 1 )
      {
	real initval = 0.0;
	fread(&initval, sizeof(real), 1, fp);
	element->update(initval);
      }

      // Read element values
      for (unsigned int node = 0; node <= order; node++)
      {
	real value = 0.0;
	fread(&value, sizeof(real), 1, fp);
	element->update(node, value);
      }
    }
  }

  // Read size of block (number of bytes)
  fwrite(&size, sizeof(unsigned int), 1, fp);
}
//-----------------------------------------------------------------------------
unsigned int ElementTmpFile::bytes(const ElementBlock& block)
{
  // Note that this function is different from ElementBlock::bytes().
  // This function needs to return the exact amount of bytes the block
  // will occupy within the binary file, whereas ElementBlock::bytes()
  // needs to return the number of bytes the block occupies in memory
  // (and only approximately).

  unsigned int sum = 0;

  // Storing size
  sum += sizeof(unsigned int);

  // Storing the interval
  sum += 2*sizeof(real);

  // Storing the number of components
  sum += sizeof(unsigned int);

  for (unsigned int i = 0; i < block.size(); i++)
  {
    const NewArray<Element*>& elements = block.components[i].elements;

    // Storing the number of elements
    sum += sizeof(unsigned int);

    typedef NewArray<Element*>::const_iterator ElementIterator;
    for (ElementIterator e = elements.begin(); e != elements.end(); ++e)
    {
      // Storing the type
      sum += sizeof(unsigned int);

      // Storing the order
      sum += sizeof(unsigned int);

      // Storing the time span
      sum += 2*sizeof(unsigned int);

      // Storing the values
      sum += ((*e)->order() + 1) * sizeof(real);
    }
  }

  // Storing size
  sum += sizeof(unsigned int);

  return sum;
}
//-----------------------------------------------------------------------------
