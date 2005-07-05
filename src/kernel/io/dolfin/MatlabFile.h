// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-07-15
// Last changed: 2005

#ifndef __MATLAB_FILE_H
#define __MATLAB_FILE_H

#include <dolfin/constants.h>
#include "MFile.h"

namespace dolfin {
  
  class Matrix;
  
  class MatlabFile : public MFile {
  public:
    
    MatlabFile(const std::string filename);
    ~MatlabFile();

    // Input
    
    // Output

    void operator<< (Matrix& A);
        
  };
  
}

#endif
