// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-07-15
// Last changed: 2005

#ifndef __PYTHON_FILE_H
#define __PYTHON_FILE_H

#include <dolfin/constants.h>
#include "GenericFile.h"

namespace dolfin
{
  
  // Represents input/output of data in a format readable by Python
  // (Numeric). The data is written to several files (the variable
  // name is appended to the base file name) to enable incremental
  // output in an efficient way.

  class PythonFile : public GenericFile
  {
  public:
    
    PythonFile(const std::string filename);
    virtual ~PythonFile();

    // Input
    
    // Output
    
    void operator<< (Sample& sample);

    std::string filename_t, filename_u, filename_k, filename_r;

  };
  
}

#endif
