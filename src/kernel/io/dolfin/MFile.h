// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#ifndef __M_FILE_H
#define __M_FILE_H

#include <dolfin/constants.h>
#include "GenericFile.h"

namespace dolfin
{
  
  class MFile : public GenericFile
  {
  public:
    
    MFile(const std::string filename);
    virtual ~MFile();

    // Input
    
    // Output
    
    void operator<< (Vector& x);
    virtual void operator<< (Matrix& A) = 0;
    void operator<< (Mesh& mesh);
    void operator<< (Function& u);
    void operator<< (Sample& sample);

  };
  
}

#endif
