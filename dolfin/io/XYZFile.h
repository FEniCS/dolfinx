// Copyright (C) 2005-2007 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Nuno Lopes 2008
// Modified by Anders Logg 2011
//
// First added:  2008-07-02
// Last changed: 2011-11-21

#ifndef __XYZ_FILE_H
#define __XYZ_FILE_H

#include <string>
#include "GenericFile.h"

namespace dolfin
{

  class Function;

  /// Simple and light file format for use with Xd3d.

  class XYZFile : public GenericFile
  {
  public:

    /// Simple and light file format for use with Xd3d. Supports
    /// scalar solution on 2D convex domains. The files only have a
    /// list of xyz coordinates 'x y u(x,y)=z'

    explicit XYZFile(const std::string filename);
    ~XYZFile();

    void operator<< (const Function& u);

  private:

    void results_write(const Function& u) const;
    void xyz_name_update();

    template<typename T>
    void mesh_function_write(T& meshfunction);

    // raw filename
    std::string xyz_filename;

  };

}

#endif
