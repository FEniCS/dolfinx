// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __M_FILE_H
#define __M_FILE_H

#include <dolfin/constants.h>
#include "GenericFile.h"

namespace dolfin {
  
  class Vector;
  class Matrix;
  class Grid;
  class Function;
  
  class MFile : public GenericFile {
  public:
    
    MFile(const std::string filename) : GenericFile(filename) {};
    
    // Input
    
    void operator>> (Vector& x);
    void operator>> (Matrix& A);
    void operator>> (Grid& grid);
    void operator>> (Function& u);
    
    // Output
    
    void operator<< (Vector& x);
    virtual void operator<< (Matrix& A) = 0;
    void operator<< (Grid& grid);
    void operator<< (Function& u);
        
  };
  
}

#endif
