// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __OPEN_DX_FILE_H
#define __OPEN_DX_FILE_H

#include <string>
#include <dolfin/constants.h>
#include <dolfin/GenericFile.h>

namespace dolfin {
  
  class Mesh;
  class Function;
  
  class OpenDXFile : public GenericFile {
  public:
    
    OpenDXFile(const std::string filename);
    ~OpenDXFile();
    
    // Input
    
    // Output
    
    void operator<< (Mesh& mesh);
    void operator<< (Function& u);
    
  };
  
}

#endif
