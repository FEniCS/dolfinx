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
// Last changed: 2012-07-29

#ifndef __DOLFIN_HDF5FILE_H
#define __DOLFIN_HDF5FILE_H

#include <string>
#include <utility>
#include "dolfin/common/types.h"
#include "GenericFile.h"

#define HDF5_FAIL -1

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
    void operator<< (const GenericVector& output);

    /// Read vector from file
    void operator>> (GenericVector& input);

    /// Write Mesh to file
    void operator<< (const Mesh& mesh);

  private:

    // Create an empty file (truncate existing)
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
    void write(T& data, const std::pair<uint,uint>& range,
               const std::string& dataset_name, const int h5type,
               const uint width) const;

    // Add a string attribute to a dataset
    void add_attribute(const std::string& dataset_name,
		       const std::string& attribute_name,
		       const std::string& attribute_value);
      
    // Friend
    friend class XDMFFile;
  };

}
#endif
