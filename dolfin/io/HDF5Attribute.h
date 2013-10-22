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
// Last changed: 2013-10-22

#ifndef __DOLFIN_HDF5ATTRIBUTE_H
#define __DOLFIN_HDF5ATTRIBUTE_H

#ifdef HAS_HDF5

#include <string>
#include <vector>

#include "HDF5Interface.h"

namespace dolfin
{

  class HDF5Attribute 
  {
  public:

    /// Constructor
    HDF5Attribute(hid_t hdf5_file_id,
                std::string dataset_name,
                std::string attribute_name) 
    : hdf5_file_id(hdf5_file_id),
      dataset_name(dataset_name),
      attribute_name(attribute_name)
    {
    }

    // Copy Constructor
    HDF5Attribute(const HDF5Attribute& hattr)
    {
      hdf5_file_id = hattr.hdf5_file_id;
      dataset_name = hattr.dataset_name;
      attribute_name = hattr.attribute_name;
    }

    /// Destructor
    ~HDF5Attribute()
    {
    }
    
    /// Set the value of the attribute in the HDF5 file
    const HDF5Attribute operator=(double& rhs);

    /// Set the value of the attribute in the HDF5 file
    const HDF5Attribute operator=(std::vector<double>& rhs);

    /// Set the value of the attribute in the HDF5 file
    const HDF5Attribute operator=(std::string& rhs);

    /// Get the value of the attribute in the HDF5 file
    /// as a string representation
    std::string str();

  private:

    hid_t hdf5_file_id;
    std::string dataset_name;
    std::string attribute_name;

    // Set the value of an attribute in the HDF5 file
    template <typename T>
    void set_value(T& value);

  };
}

#endif
#endif
