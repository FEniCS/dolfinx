#include <dolfin/File.h>
#include "GenericFile.h"
#include "XMLFile.h"

// FIXME
#include <string>
#include <iostream>

using namespace dolfin;

//-----------------------------------------------------------------------------
File::File(const std::string& filename)
{
  if ( filename.rfind(".xml") != filename.npos )
	 file = new XMLFile(filename);
  else if ( filename.rfind(".xml.gz") != filename.npos )
	 file = new XMLFile(filename);
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
void File::operator>>(SparseMatrix& sparseMatrix)
{
  *file >> sparseMatrix;
}
//-----------------------------------------------------------------------------
void File::operator>>(Grid& grid)
{
  *file >> grid;
}
//-----------------------------------------------------------------------------
void File::operator<<(const Vector& vector)
{
  *file << vector;
}
//-----------------------------------------------------------------------------
void File::operator<<(const SparseMatrix& sparseMatrix)
{
  *file << sparseMatrix;
}
//-----------------------------------------------------------------------------
void File::operator<<(const Grid& grid)
{
  *file << grid;
}
//-----------------------------------------------------------------------------
