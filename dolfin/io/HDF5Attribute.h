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
// Last changed: 2014-11-24

#ifndef __DOLFIN_HDF5ATTRIBUTE_H
#define __DOLFIN_HDF5ATTRIBUTE_H

#ifdef HAS_HDF5

#include <string>
#include <vector>

#include<dolfin/common/Array.h>
#include "HDF5Interface.h"

namespace dolfin
{

  /// HDF5Attribute gives access to the attributes of a dataset
  /// via set() and get() methods

  class HDF5Attribute
  {
  public:

    // FIXME: Check validity of file and dataset

    /// Constructor
    HDF5Attribute(const hid_t hdf5_file_id, std::string dataset_name)
      : hdf5_file_id(hdf5_file_id), dataset_name(dataset_name) {}

    /// Destructor
    ~HDF5Attribute() {}

    /// Check for the existence of an attribute on a dataset
    bool exists(const std::string attribute_name) const;

    /// Set the value of a double attribute in the HDF5 file
    void set(const std::string attribute_name, const double value);

    /// Set the value of a double attribute in the HDF5 file
    void set(const std::string attribute_name, const std::size_t value);

    /// Set the value of an array of float attribute in the HDF5 file
    void set(const std::string attribute_name,
             const std::vector<double>& value);

    /// Set the value of an array of float attribute in the HDF5 file
    void set(const std::string attribute_name,
             const std::vector<std::size_t>& value);

    /// Set the value of a string attribute in the HDF5 file
    void set(const std::string attribute_name, const std::string value);

    /// Set the value of a double attribute in the HDF5 file
    void get(const std::string attribute_name, double& value) const;

    /// Get the value of a vector double attribute in the HDF5 file
    void get(const std::string attribute_name,
             std::vector<double>& value) const;

    /// Set the value of a double attribute in the HDF5 file
    void get(const std::string attribute_name, std::size_t& value) const;

    /// Get the value of a vector double attribute in the HDF5 file
    void get(const std::string attribute_name,
             std::vector<std::size_t>& value) const;

    /// Get the value of an attribute in the HDF5 file as a string
    void get(const std::string attribute_name, std::string& value) const;

    /// Get the value of the attribute in the HDF5 file
    /// as a string representation
    const std::string str(const std::string attribute_name) const;

    /// Get the type of the attribute "string", "float", "int"
    /// "vectorfloat", "vectorint" or "unsupported"
    const std::string type_str(const std::string attribute_name) const;

    /// Get the names of all the attributes on this dataset
    const std::string str() const;

    /// Get the names of all the attributes on this dataset as a
    /// std::vector<std::string>
    const std::vector<std::string> list_attributes() const;

  private:

    const hid_t hdf5_file_id;
    const std::string dataset_name;

    // Set the value of an attribute in the HDF5 file
    template <typename T>
    void set_value(const std::string attribute_name, const T& value);

    // Get the value of an attribute in the HDF5 file
    template <typename T>
    void get_value(const std::string attribute_name, T& value) const;

    template <typename T>
    const std::string
      vector_to_string(const std::vector<T>& vector_value) const;

  };
}

#endif
#endif
