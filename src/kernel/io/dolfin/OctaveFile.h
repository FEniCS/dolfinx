// Copyright (C) 2003-2006 Johan Hoffman and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-07-15
// Last changed: 2005-05-07

#ifndef __OCTAVE_FILE_H
#define __OCTAVE_FILE_H

#include <dolfin/constants.h>
#include <dolfin/Matrix.h>
#include <dolfin/MFile.h>

namespace dolfin
{

  class OctaveFile : public MFile
  {
  public:
    
    OctaveFile(const std::string filename);
    ~OctaveFile();

    // Input
    
    // Output
    
    void operator<< (Matrix& A);
    
  };
  
}

#endif
