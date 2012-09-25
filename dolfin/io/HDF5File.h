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
// Last changed: 2012-09-25

#ifndef __DOLFIN_HDF5FILE_H
#define __DOLFIN_HDF5FILE_H

#ifdef HAS_HDF5

#include <string>
#include <utility>
#include <vector>
#include "dolfin/common/types.h"
#include "GenericFile.h"

namespace dolfin
{

  class Function;
  class GenericVector;
  class Mesh;

  class HDF5File: public GenericFile
  {
  public:

    /// Constructor
    HDF5File(const std::string filename);

    /// Destructor
    ~HDF5File();

    /// Write vector to file
    /// saves into HDF5 folder 'Vector'
    /// multiple calls will save in the same file
    /// with incrementing dataset names
    void operator<< (const GenericVector& x);

    /// Read vector from file
    /// looks in HDF5 folder 'Vector' for dataset 0
    void operator>> (GenericVector& x);

    /// Write Mesh to file
    void operator<< (const Mesh& mesh);

    /// Write Mesh to file. 'true_topology_indices' indicates
    /// whether the true vertex indices should be used for the connectivity
    /// or the position of the vertex in the list. The latter is required
    /// for visualisation and the former for reading a Mesh from file.
    void write_mesh(const Mesh& mesh, bool true_topology_indices=true);

    /// Read Mesh from file
    void operator>> (Mesh& mesh);

  private:

    // Friend
    friend class XDMFFile;

    // Create an empty file (truncate if existing)
    void create();

    // Write data to existing HDF file contiguously from each process,
    // the range being set by the data size
    // width: is the width of the data item (dim 1, e.g. 3 for x, y, z data)
    template <typename T>
    void write(const std::string dataset_name,
               const std::vector<T>& data,
               const uint width);

    // Check if dataset exists in this file
    bool dataset_exists(const std::string &dataset_name);
    
    // Search through list of datasets for one beginning with search_term
    std::string search_list(std::vector<std::string> &list_of_strings, 
                            const std::string &search_term) const;

    // Generate HDF5 dataset names for mesh topology and coordinates
    std::string mesh_coords_dataset_name(const Mesh& mesh) const;
    std::string mesh_index_dataset_name(const Mesh& mesh) const;
    std::string mesh_topology_dataset_name(const Mesh& mesh) const;

  };

}
#endif
#endif
