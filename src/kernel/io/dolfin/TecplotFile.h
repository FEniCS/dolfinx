// Copyright (C) 2004 Harald Svensson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2005.

#ifndef __TECPLOT_FILE_H
#define __TECPLOT_FILE_H

#include <string>
#include <dolfin/constants.h>
#include <dolfin/GenericFile.h>

namespace dolfin {

  class Mesh;
  class Function;

  class TecplotFile : public GenericFile {
  public:

    TecplotFile(const std::string filename);
    ~TecplotFile();

    // Input

    // Output

    void operator<< (Mesh& mesh);
    void operator<< (Function& u);
    void operator<< (Function::Vector& u);

  };
  
}

#endif
