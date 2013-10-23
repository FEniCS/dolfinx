// Copyright (C) 2013 Chris N. Richardson
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
// First added:  2013-10-16
// Last changed: 2013-10-23

#ifndef __DOLFIN_HDF5ATTRIBUTE_H
#define __DOLFIN_HDF5ATTRIBUTE_H

#ifdef HAS_HDF5

#include <string>
#include <vector>

#include<dolfin/common/Array.h>

#include "HDF5Interface.h"

namespace dolfin
{


  class HDF5Attribute 
  {
  public:

    /// Constructor
    HDF5Attribute(hid_t hdf5_file_id,
                  std::string dataset_name)
    : hdf5_file_id(hdf5_file_id),
      dataset_name(dataset_name)
    {
    }

    /// Destructor
    ~HDF5Attribute()
    {
    }
    
    /// Set the value of a double attribute in the HDF5 file
    void set(const std::string attribute_name, const double value);

    /// Set the value of an array of float attribute in the HDF5 file
    void set(const std::string attribute_name, const Array<double>& value);

    /// Set the value of a string attribute in the HDF5 file
    void set(const std::string attribute_name, const std::string value);

    /// Set the value of a double attribute in the HDF5 file
    void get(const std::string attribute_name, double& value) const;

    /// Get the value of a vector double attribute in the HDF5 file
    void get(const std::string attribute_name, 
             std::vector<double>& value) const;

    /// Get the value of an attribute in the HDF5 file as a string
    void get(const std::string attribute_name, std::string& value) const;

    /// Get the value of the attribute in the HDF5 file
    /// as a string representation
    const std::string str(const std::string attribute_name) const;

    /// Get the type of the attribute "string", "float", "vector"
    /// or "unsupported"
    const std::string type(const std::string attribute_name) const;

  private:

    hid_t hdf5_file_id;
    std::string dataset_name;

    // Set the value of an attribute in the HDF5 file
    template <typename T>
    void set_value(const std::string attribute_name, const T& value);

    // Get the value of an attribute in the HDF5 file
    template <typename T>
    void get_value(const std::string attribute_name, T& value) const;

  };
}

#endif
#endif
