// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/LoggerMacros.h>
#include <dolfin/Buffer.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Buffer::Buffer()
{
  buffer = 0;
  clear();
}
//-----------------------------------------------------------------------------
Buffer::Buffer(int lines, int cols)
{
  buffer = 0;
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
  
  buffer = new (char *)[lines];
  for (int i = 0; i < lines; i++) {
    buffer[i] = new char[cols+1];
    buffer[i][0] = '\0';
  }
}
//-----------------------------------------------------------------------------
int Buffer::size()
{
  if ( full )
    return lines;
  
  return last - first + 1;
}
//-----------------------------------------------------------------------------
void Buffer::add(const char* msg)
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
}
//-----------------------------------------------------------------------------
const char* Buffer::get(int line)
{
  if ( line < 0 )
    dolfin_error("Line number must be non-negative.");
  if ( line >= lines )
    dolfin_error1("Line number too large: %d.", line);

  return buffer[(first + line) % lines];
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

  lines = 0;
  cols = 0;
  first = 0;
  last = 0;
  full = false;
}
//-----------------------------------------------------------------------------
