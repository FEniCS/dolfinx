// Copyright (C) 2004 Harald Svensson.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-03-30
// Last changed: 2004

#ifndef __GID_FILE_H
#define __GID_FILE_H

#include <string>
#include <dolfin/constants.h>
#include <dolfin/GenericFile.h>

namespace dolfin
{

  class Mesh;

  class GiDFile : public GenericFile
  {
  public:

    GiDFile(const std::string filename);
    ~GiDFile();

    // Input

    // Output

    void operator<< (Mesh& mesh);

  };
  
}

#endif
