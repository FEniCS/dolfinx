// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __MATLAB_FILE_H
#define __MATLAB_FILE_H

#include <dolfin/constants.h>
#include "MFile.h"

namespace dolfin {
  
  class Vector;
  class Matrix;
  class Grid;
  class Function;
  
  class MatlabFile : public MFile {
  public:
    
    MatlabFile(const std::string filename) : MFile(filename) {};
    
    // Input
    
    // Output

    void operator<< (Matrix& A);
        
  };
  
}

#endif
