#include "GenericFile.h"

// FIXME, this should not be here
#include <iostream>

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericFile::GenericFile(const std::string filename)
{
  this->filename = filename;
}
//-----------------------------------------------------------------------------
GenericFile::~GenericFile()
{

}
//-----------------------------------------------------------------------------
