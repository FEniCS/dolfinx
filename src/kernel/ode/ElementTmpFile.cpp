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
ElementTmpFile::ElementTmpFile()
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
