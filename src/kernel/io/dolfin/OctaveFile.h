// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __OCTAVE_FILE_H
#define __OCTAVE_FILE_H

#include <dolfin/constants.h>
#include "MFile.h"

namespace dolfin {

  class Vector;
  class Matrix;
  class Mesh;
  class Function;
  
  class OctaveFile : public MFile {
  public:
    
    OctaveFile(const std::string filename) : MFile(filename) {};
    
    // Input
    
    // Output
    
    void operator<< (Matrix& A);
    
  };
  
}

#endif
