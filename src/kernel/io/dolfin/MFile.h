// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __M_FILE_H
#define __M_FILE_H

#include <dolfin/constants.h>
#include "GenericFile.h"

namespace dolfin {
  
  class Vector;
  class Matrix;
  class Mesh;
  class NewSample;
  
  class MFile : public GenericFile {
  public:
    
    MFile(const std::string filename);
    virtual ~MFile();

    // Input
    
    // Output
    
    void operator<< (Vector& x);
    virtual void operator<< (Matrix& A) = 0;
    void operator<< (Mesh& mesh);
    void operator<< (NewFunction& u);
    void operator<< (NewSample& sample);

  };
  
}

#endif
