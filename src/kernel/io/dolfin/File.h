// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <string>

#ifndef __FILE_H
#define __FILE_H

namespace dolfin {

  class Vector;
  class SparseMatrix;
  class Grid;
  class GenericFile;
  
  class File {
  public:

	 File(const std::string& filename);
	 ~File();

	 // Input
	 
	 void operator>> (Vector& vector);
	 void operator>> (SparseMatrix& sparseMatrix);
	 void operator>> (Grid& grid);
	 
	 // Output

	 void operator<< (const Vector& vector);
	 void operator<< (const SparseMatrix& sparseMatrix);
	 void operator<< (const Grid& grid);
	 
  private:

	 GenericFile *file;
	 
  };

}

#endif
