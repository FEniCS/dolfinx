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
// Last changed: 2012-09-17

#ifndef __DOLFIN_HDF5FILE_H
#define __DOLFIN_HDF5FILE_H

#ifdef HAS_HDF5

#include <string>
#include <utility>
#include "dolfin/common/types.h"
#include "GenericFile.h"

#define HDF5_FAIL -1
#define HDF5_MAXSTRLEN 80

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
    ///
    void operator<< (const GenericVector& output);

    /// Read vector from file
    /// looks in HDF5 folder 'Vector' for last dataset
    ///
    void operator>> (GenericVector& input);

    /// Write Mesh to file
    /// Saves into folder 'Mesh' as two datasets,
    /// 'Topology' and 'Coordinates'
    ///
    void operator<< (const Mesh& mesh);

    /// Read Mesh from file
    void operator>> (Mesh& mesh);

  private:

    // Friend
    friend class XDMFFile;

    // Create an empty file (truncate if existing)
    void create();

    // Write functions for int, double, etc. Used by XDMFFile
    void write(const double& data,
               const std::pair<uint, uint>& range,
               const std::string& dataset_name,
               const uint width);

    void write(const uint& data,
               const std::pair<uint, uint>& range,
               const std::string& dataset_name,
               const uint width);

    template <typename T>
    void write(const T& data, const std::pair<uint,uint>& range,
               const std::string& dataset_name, const int h5type,
               const uint width) const;

    template <typename T>
    void read(T& data, const std::pair<uint,uint>& range,
               const std::string& dataset_name, const int h5type,
               const uint width) const;

    // Get dimensions of 2D dataset
    std::pair<uint,uint> dataset_dimensions(const std::string& dataset_name);

    // List all datasets in a group
    std::vector<std::string> list(const std::string& group_name);

    // Check existence of dataset in file
    bool exists(const std::string& dataset_name);

    // Add/get a string attribute to/from a dataset
    void add_attribute(const std::string& dataset_name,
                       const std::string& attribute_name,
                       const std::string& attribute_value);

    std::string get_attribute(const std::string& dataset_name,
                              const std::string& attribute_name);

    // Generate HDF5 dataset names for mesh topology and coordinates
    std::string mesh_coords_dataset_name(const Mesh& mesh);
    std::string mesh_index_dataset_name(const Mesh& mesh);
    std::string mesh_topo_dataset_name(const Mesh& mesh);

  };

}
#endif
#endif
