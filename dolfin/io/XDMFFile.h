// Copyright (C) 2012 Chris N. Richardson
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
// Modified by Garth N. Wells, 2012
//
// First added:  2012-05-22
// Last changed: 2012-08-02

#ifndef __DOLFIN_XDMFFILE_H
#define __DOLFIN_XDMFFILE_H

#ifdef HAS_HDF5

#include <string>
#include <utility>
#include <boost/shared_ptr.hpp>
#include "GenericFile.h"

namespace pugi
{
  class xml_node;
}

namespace dolfin
{

  // Forward declarations
  class Function;
  class Mesh;

  class XDMFFile: public GenericFile
  {
  public:

    /// Constructor
    XDMFFile(const std::string filename);

    /// Destructor
    ~XDMFFile();

    /// Save Mesh to file
    /// data in .h5 (HDF5), XML description in .xdmf
    void operator<<(const Mesh& mesh);

    /// Save Function to file
    /// data in .h5 (HDF5), XML description in .xdmf
    void operator<<(const Function& u);

    /// Save Function + time stamp to file
    /// data in .h5 (HDF5), XML description in .xdmf
    void operator<<(const std::pair<const Function*, double> ut);

  private:

    // Write out a cell-centred function
    void write_cell_function(const std::pair<const Function*, double> ut);

    // Change file suffix from .xdmf to .h5
    std::string HDF5Filename() const;

  };

}
#endif
#endif
