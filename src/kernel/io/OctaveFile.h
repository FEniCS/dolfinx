// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __OCTAVE_FILE_H
#define __OCTAVE_FILE_H

#include <dolfin/constants.h>
#include "GenericFile.h"

namespace dolfin {

  class Vector;
  class Matrix;
  class Grid;
  class Function;
  
  class OctaveFile : public GenericFile {
  public:

	 OctaveFile(const std::string filename) : GenericFile(filename) {};
	 
	 // Input
	 
	 void operator>> (Vector& x);
	 void operator>> (Matrix& A);
	 void operator>> (Grid& grid);
	 void operator>> (Function& u);	 
	 
	 // Output
	 
	 void operator<< (const Vector& x);
	 void operator<< (const Matrix& A);
	 void operator<< (const Grid& grid);
	 void operator<< (const Function& u);

  };
  
}

#endif
