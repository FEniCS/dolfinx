// Copyright (C) 2003-2008 Johan Hoffman and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-07-15
// Last changed: 2008-03-29

#ifndef __M_FILE_H
#define __M_FILE_H

#include <dolfin/main/constants.h>
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

    void operator<< (GenericVector& x);
    virtual void operator<< (Matrix& A) = 0;
    void operator<< (Mesh& mesh);
    void operator<< (Function& u);
    void operator<< (Sample& sample);

  };
  
}

#endif
