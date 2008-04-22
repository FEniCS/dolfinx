// Copyright (C) 2003-2006 Johan Hoffman and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-07-15
// Last changed: 2006-05-07

#ifndef __MATLAB_FILE_H
#define __MATLAB_FILE_H

#include <dolfin/common/types.h>
#include <dolfin/la/GenericMatrix.h>
#include "MFile.h"

namespace dolfin
{
  
  class MatlabFile : public MFile
  {
  public:
    
    MatlabFile(const std::string filename);
    ~MatlabFile();

    // Input
    
    // Output

    void operator<< (GenericMatrix& A);
        
  };
  
}

#endif
