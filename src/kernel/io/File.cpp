#include <dolfin/File.h>
#include "GenericFile.h"
#include "DolfinFile.h"

// FIXME
#include <iostream>

using namespace dolfin;

//-----------------------------------------------------------------------------
File::File(const std::string& filename)
{
  if ( filename.rfind(".dolfin") != filename.npos )
	 file = new DolfinFile(filename);
  else{
	 cout << "Unknown type for file " << filename << endl;
	 exit(1);
  }
}
//-----------------------------------------------------------------------------
File::~File()
{
  delete file;
}
//-----------------------------------------------------------------------------
void File::operator>>(Vector& vector)
{
  *file >> vector;
}
//-----------------------------------------------------------------------------
void File::operator<<(const Vector &vector)
{
  *file << vector;
}
//-----------------------------------------------------------------------------
