// Copyright (C) 2004 Harald Svensson.
// Licensed under the GNU GPL Version 2.

#ifndef __GID_FILE_H
#define __GID_FILE_H

#include <string>
#include <dolfin/constants.h>
#include <dolfin/GenericFile.h>

namespace dolfin {

  class Mesh;
  class Function;

  class GiDFile : public GenericFile {
  public:

    GiDFile(const std::string filename);
    ~GiDFile();

    // Input

    // Output

    void operator<< (Mesh& mesh);
    void operator<< (Function& u);
    void operator<< (Function::Vector& u);

  };
  
}

#endif
