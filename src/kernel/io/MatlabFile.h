// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __MATLAB_FILE_H
#define __MATLAB_FILE_H

#include <dolfin/constants.h>
#include "GenericFile.h"

namespace dolfin {

  class Vector;
  class Matrix;
  
  class MatlabFile : public GenericFile {
  public:

	 MatlabFile(const std::string filename) : GenericFile(filename) {};
	 
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

	 // Friends
	 friend class OctaveFile;
	 
  protected:

	 // These functions can be called also from within OctaveFile

	 static void writeVector  (const Vector& x,
										const std::string& filename,
										const std::string& name,
										const std::string& label);
	 
	 static void writeMatrix  (const Matrix& A,
										const std::string& filename,
										const std::string& name,
										const std::string& label);
	 
	 static void writeGrid    (const Grid& grid,
										const std::string& filename,
										const std::string& name,
										const std::string& label);
	 
	 static void writeFunction(const Function& u,
										const std::string& filename,
										const std::string& name,
										const std::string& label);
	 
  };
  
}

#endif
