// Copyright (C) 2013 Chris N Richardson
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
//
// First added:  2012-06-01
// Last changed: 2013-10-22

#ifdef HAS_HDF5

#include "HDF5Attribute.h"
#include "HDF5Interface.h"

using namespace dolfin;

template <typename T>
void HDF5Attribute::set_value(T& attribute_value)
{

  if(!HDF5Interface::has_dataset(hdf5_file_id, dataset_name))
  {
    dolfin_error("HDF5File.cpp", 
                 "set attribute on dataset",
                 "Dataset does not exist");
  }
  
  if(HDF5Interface::has_attribute(hdf5_file_id, dataset_name, 
                                  attribute_name))
  {
    HDF5Interface::delete_attribute(hdf5_file_id, dataset_name, 
                                    attribute_name);
  }
  
  HDF5Interface::add_attribute(hdf5_file_id, dataset_name, 
                               attribute_name, attribute_value);
}


const HDF5Attribute HDF5Attribute::operator=(double& rhs)
{
  set_value(rhs);
  return *this;
}

const HDF5Attribute HDF5Attribute::operator=(std::vector<double>& rhs)
{
  set_value(rhs);
  return *this;
}

const HDF5Attribute HDF5Attribute::operator=(std::string& rhs)
{
  set_value(rhs);
  return *this;
}

std::string HDF5Attribute::str()
{
  
  if(!HDF5Interface::has_dataset(hdf5_file_id, dataset_name))
  {
    dolfin_error("HDF5Attribute.cpp", 
                 "get attribute of dataset",
                 "Dataset does not exist");
  }
  
  if(!HDF5Interface::has_attribute(hdf5_file_id, dataset_name, 
                                  attribute_name))
  {
    dolfin_error("HDF5Attribute.cpp",
                 "get attribute of dataset",
                 "Attribute does not exist");
  }

  return HDF5Interface::get_attribute_string(hdf5_file_id, 
                              dataset_name, attribute_name);
  
}



#endif
