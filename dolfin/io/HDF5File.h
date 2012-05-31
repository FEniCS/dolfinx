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
//
// First added:  2012-05-22
// Last changed: 2012-05-22

#ifndef __HDF5FILE_H
#define __HDF5FILE_H

#include <map>
#include <ostream>
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include "GenericFile.h"

namespace dolfin
{

  class Function;
  class GenericVector;
  class LocalMeshData;
  class Mesh;
  class Parameters;
  template<typename T> class Array;
  template<typename T> class MeshFunction;
  template<typename T> class MeshValueCollection;

  class HDF5File: public GenericFile
  {
  public:

    /// Constructor
    HDF5File(const std::string filename);

    ~HDF5File();

    // Vector
    void operator<< (const GenericVector& output);
    void operator>> (GenericVector& input);

    // Function
    void operator<< (const Function& u);

    // Mesh
    void operator<< (const Mesh& mesh);

    //write functions for int,double, etc.
    void write(const double& data,
	       const std::pair<uint,uint>& range,
	       const std::string& dataset_name,
	       const uint width);

    void write(const uint& data,
	       const std::pair<uint,uint>& range,
	       const std::string& dataset_name,
	       const uint width);

    template <typename T>
    void write(T& data, 
	       const std::pair<uint,uint>& range,
	       const std::string& dataset_name,
	       int h5type,
	       uint width);

  private:

    boost::shared_ptr<std::ostream> outstream;

  };

}
#endif
