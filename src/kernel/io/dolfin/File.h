// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <string>

#ifndef __FILE_H
#define __FILE_H

namespace dolfin {

  class Vector;
  class Matrix;
  class Grid;
  class GenericFile;
  
  class File {
  public:

	 File(const std::string& filename);
	 ~File();

	 // Input
	 
	 void operator>> (Vector& x);
	 void operator>> (Matrix& A);
	 void operator>> (Grid& grid);
	 
	 // Output

	 void operator<< (const Vector& x);
	 void operator<< (const Matrix& A);
	 void operator<< (const Grid& grid);
	 
  private:

	 GenericFile* file;
	 
  };

}

#endif
