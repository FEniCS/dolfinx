// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __OCTAVE_FILE_H
#define __OCTAVE_FILE_H

#include <dolfin/constants.h>
#include "MFile.h"

namespace dolfin {

  class Matrix;
  
  class OctaveFile : public MFile {
  public:
    
    OctaveFile(const std::string filename);
    ~OctaveFile();

    // Input
    
    // Output
    
    void operator<< (Matrix& A);
    
  };
  
}

#endif
