// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// FIXME: Use streams instead of stdio
#include <stdio.h>

#include "GenericFile.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericFile::GenericFile(const std::string filename)
{
  this->filename = filename;

  opened_read = false;
  opened_write = false;
  
}
//-----------------------------------------------------------------------------
GenericFile::~GenericFile()
{

}
//-----------------------------------------------------------------------------
void GenericFile::read()
{
  opened_read = true;
}
//-----------------------------------------------------------------------------
void GenericFile::write()
{
  if ( !opened_write ) {
	 FILE* fp = fopen(filename.c_str(), "w");
	 fclose(fp);
  }
  
  opened_write = true;
}
//-----------------------------------------------------------------------------
