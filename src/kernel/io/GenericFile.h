// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GENERIC_FILE_H
#define __GENERIC_FILE_H

#include <string>
#include <fstream>


namespace dolfin {

  class Vector;
  class SparseMatrix;
  class Grid;
  
  class GenericFile {
  public:
	 
	 GenericFile(const std::string filename);
	 virtual ~GenericFile();
	 
	 // Input
	 
	 virtual void operator>> (Vector& vector) = 0;
	 virtual void operator>> (SparseMatrix& sparseMatrix) = 0;
	 virtual void operator>> (Grid& grid) = 0;
	 
	 // Output
	 
	 virtual void operator<< (const Vector& vector) = 0;
	 virtual void operator<< (const SparseMatrix& sparseMatrix) = 0;
	 virtual void operator<< (const Grid& grid) = 0;
	 
  protected:

	 std::string filename;
	 
  };

}

#endif
