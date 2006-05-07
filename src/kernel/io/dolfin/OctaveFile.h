// Copyright (C) 2003-2006 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-07-15
// Last changed: 2005-05-07

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
    
#ifdef HAVE_PETSC_H
    void operator<< (Matrix& A);
#endif
    
  };
  
}

#endif
