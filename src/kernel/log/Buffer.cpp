// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-03-26
// Last changed: 2005

#include <dolfin/log.h>
#include <dolfin/Buffer.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Buffer::Buffer()
{
  buffer = 0;
  types = 0;
  levels = 0;
  clear();
}
//-----------------------------------------------------------------------------
Buffer::Buffer(int lines, int cols)
{
  buffer = 0;
  types = 0;
  levels = 0;
  init(lines, cols);
}
//-----------------------------------------------------------------------------
Buffer::~Buffer()
{
  clear();
}
//-----------------------------------------------------------------------------
void Buffer::init(int lines, int cols)
{
  if ( lines <= 0 )
    dolfin_error("Number of lines must be positive.");
  if ( cols <= 0 )
    dolfin_error("Number of columns must be positive.");
  
  clear();
  
  this->lines = lines;
  this->cols = cols;
  
  first = 0;
  last = -1;
  full = false;
  
  buffer = new char *[lines];
  for (int i = 0; i < lines; i++) {
    buffer[i] = new char[cols+1];
    buffer[i][0] = '\0';
  }
  
  types = new Type[lines];
  for (int i = 0; i < lines; i++)
    types[i] = info;

  levels = new int[lines];
  for (int i = 0; i < lines; i++)
    levels[i] = 0;
}
//-----------------------------------------------------------------------------
int Buffer::size() const
{
  if ( full )
    return lines;
  
  return last - first + 1;
}
//-----------------------------------------------------------------------------
void Buffer::add(const char* msg, Type type, int level)
{
  // Step last
  last++;
  if ( last == lines ) {
    last = 0;
    full = true;
  }
  
  // Step first
  if ( full ) {
    first++;
    if ( first == lines )
      first = 0;
  }

  // Copy the message into the buffer
  int i = 0;
  for (i = 0; msg[i] && i < cols; i++) {
    buffer[last][i] = msg[i];
  }
  buffer[last][i] = '\0';

  // Save the type
  types[last] = type;

  // Save the level
  levels[last] = level;
}
//-----------------------------------------------------------------------------
const char* Buffer::get(int line) const
{
  if ( line < 0 )
    dolfin_error("Line number must be non-negative.");
  if ( line >= lines )
    dolfin_error1("Line number too large: %d.", line);

  return buffer[(first + line) % lines];
}
//-----------------------------------------------------------------------------
Buffer::Type Buffer::type(int line) const
{
  if ( line < 0 )
    dolfin_error("Line number must be non-negative.");
  if ( line >= lines )
    dolfin_error1("Line number too large: %d.", line);

  return types[(first + line) % lines];
}
//-----------------------------------------------------------------------------
int Buffer::level(int line) const
{
  if ( line < 0 )
    dolfin_error("Line number must be non-negative.");
  if ( line >= lines )
    dolfin_error1("Line number too large: %d.", line);

  return levels[(first + line) % lines];
}
//-----------------------------------------------------------------------------
void Buffer::clear()
{
  if ( buffer ) {
    for (int i = 0; i < lines; i++)
      delete [] buffer[i];
    delete [] buffer;
  }
  buffer = 0;

  if ( types )
    delete [] types;
  types = 0;

  if ( levels )
    delete [] levels;
  levels = 0;

  lines = 0;
  cols = 0;
  first = 0;
  last = 0;
  full = false;
}
//-----------------------------------------------------------------------------
