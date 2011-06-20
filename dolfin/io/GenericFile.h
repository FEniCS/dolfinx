// Copyright (C) 2003-2008 Johan Hoffman and Anders Logg
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
// Modified by Ola Skavhaug 2009.
//
// First added:  2003-07-15
// Last changed: 2009-03-16

#ifndef __GENERIC_FILE_H
#define __GENERIC_FILE_H

#include <map>
#include <string>
#include <utility>
#include <vector>
#include "dolfin/common/types.h"

namespace dolfin
{

  class Function;
  class FunctionPlotData;
  class GenericMatrix;
  class GenericVector;
  class LocalMeshData;
  class Mesh;
  template <class T> class MeshFunction;
  class Parameters;

  class GenericFile
  {
  public:

    /// Constructor
    GenericFile(const std::string filename);

    // Input
    virtual void operator>> (Mesh& mesh) const;
    virtual void operator>> (GenericVector& x) const;
    virtual void operator>> (GenericMatrix& A);
    virtual void operator>> (LocalMeshData& data);
    virtual void operator>> (MeshFunction<int>& meshfunction) const;
    virtual void operator>> (MeshFunction<unsigned int>& meshfunction) const;
    virtual void operator>> (MeshFunction<double>& meshfunction) const;
    virtual void operator>> (MeshFunction<bool>& meshfunction) const;
    virtual void operator>> (Parameters& parameters) const;
    virtual void operator>> (FunctionPlotData& data);
    virtual void operator>> (std::vector<int>& x);
    virtual void operator>> (std::vector<uint>& x);
    virtual void operator>> (std::vector<double>& x);
    virtual void operator>> (std::map<uint, int>& map);
    virtual void operator>> (std::map<uint, uint>& map);
    virtual void operator>> (std::map<uint, double>& map);
    virtual void operator>> (std::map<uint, std::vector<int> >& array_map);
    virtual void operator>> (std::map<uint, std::vector<uint> >& array_map);
    virtual void operator>> (std::map<uint, std::vector<double> >& array_map);

    // Output
    virtual void operator<< (const GenericVector& x);
    virtual void operator<< (const GenericMatrix& A);
    virtual void operator<< (const Mesh& mesh);
    virtual void operator<< (const LocalMeshData& data);
    virtual void operator<< (const MeshFunction<int>& meshfunction);
    virtual void operator<< (const MeshFunction<unsigned int>& meshfunction);
    virtual void operator<< (const MeshFunction<double>& meshfunction);
    virtual void operator<< (const MeshFunction<bool>& meshfunction);
    virtual void operator<< (const Function& u);

    // Output function with time
    virtual void operator<< (const std::pair<const Function*, double> u);

    virtual void operator<< (const Parameters& parameters);
    virtual void operator<< (const FunctionPlotData& data);
    virtual void operator<< (const std::vector<int>& x);
    virtual void operator<< (const std::vector<uint>& x);
    virtual void operator<< (const std::vector<double>& x);
    virtual void operator<< (const std::map<uint, int>& map);
    virtual void operator<< (const std::map<uint, uint>& map);
    virtual void operator<< (const std::map<uint, double>& map);
    virtual void operator<< (const std::map<uint, std::vector<int> >& array_map);
    virtual void operator<< (const std::map<uint, std::vector<uint> >& array_map);
    virtual void operator<< (const std::map<uint, std::vector<double> >& array_map);

    void read();
    virtual void write();

  protected:

    void read_not_impl(const std::string object) const;
    void write_not_impl(const std::string object) const;

    std::string filename;
    std::string type;

    bool opened_read;
    bool opened_write;

    // True if we have written a header
    bool check_header;

    // Counters for the number of times various data has been written
    uint counter;
    uint counter1;
    uint counter2;

  };

}

#endif
