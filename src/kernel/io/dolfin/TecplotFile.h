// Copyright (C) 2004 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.
//
// First added:  2004
// Last changed: 2005

#ifndef __TECPLOT_FILE_H
#define __TECPLOT_FILE_H

#include <string>
#include <dolfin/constants.h>
#include <dolfin/GenericFile.h>

namespace dolfin
{

  class Mesh;

  class TecplotFile : public GenericFile
  {
  public:

    TecplotFile(const std::string filename);
    ~TecplotFile();

    // Input

    // Output
    void operator<< (Mesh& mesh);

  };
  
}

#endif
