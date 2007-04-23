// Copyright (C) 2004-2005 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2005.
//
// First added:  2004
// Last changed: 2005

#ifndef __TECPLOT_FILE_H
#define __TECPLOT_FILE_H

#include <dolfin/GenericFile.h>

namespace dolfin
{

  class TecplotFile : public GenericFile
  {
  public:

    TecplotFile(const std::string filename);
    ~TecplotFile();

    // Input

    // Output
    void operator<< (Mesh& mesh);
    void operator<< (Function& u);

  };
  
}

#endif
