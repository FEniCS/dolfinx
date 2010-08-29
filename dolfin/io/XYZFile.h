// Copyright (C) 2005-2007 Garth N.Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Nuno Lopes 2008.
//
// First added:  2008-07-02

#ifndef __XYZ_FILE_H
#define __XYZ_FILE_H

#include <string>
#include "GenericFile.h"

namespace dolfin
{

  class Function;

  class XYZFile : public GenericFile
  {
  public:

    /// Simple and light file format for use with Xd3d. Supports
    /// scalar solution on 2D convex domains. The files only have a
    /// list of xyz coordinates 'x y u(x,y)=z'

    XYZFile(const std::string filename);
    ~XYZFile();

    void operator<< (const Function& u);

  private:

    void results_write(const Function& u) const;
    void xyz_name_update(int counter);

    template<class T>
    void mesh_function_write(T& meshfunction);

    // raw filename
    std::string xyz_filename;

  };

}

#endif
