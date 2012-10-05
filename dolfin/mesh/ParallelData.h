// Copyright (C) 2011 Anders Logg and Garth N. Wells
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
// First added:  2011-01-17
// Last changed: 2011-09-27

#ifndef __PARALLEL_DATA_H
#define __PARALLEL_DATA_H

#include <map>
#include <utility>
#include <vector>
#include <boost/scoped_ptr.hpp>
#include "dolfin/common/types.h"

namespace dolfin
{

  class Mesh;
  template<typename T> class MeshFunction;

  /// This class stores auxiliary mesh data for parallel computing.

  class ParallelData
  {
  public:

    /// Constructor
    ParallelData(const Mesh& mesh);

    /// Copy constructor
    ParallelData(const ParallelData& data);

    /// Destructor
    ~ParallelData();

    //--- Data for distributed memory parallelism ---

    /// Return MeshFunction that is true for globally exterior facets,
    /// false otherwise
    MeshFunction<bool>& exterior_facet();

    /// Return MeshFunction that is true for globally exterior facets,
    /// false otherwise (const version)
    const MeshFunction<bool>& exterior_facet() const;


    //--- Data for shared memory parallelism (multicore) ---

    /// First vector is (colored entity dim - dim0 - .. -  colored entity dim).
    /// MeshFunction stores mesh entity colors and the vector<vector> is a list
    /// of all mesh entity indices of the same color,
    /// e.g. vector<vector>[col][i] is the index of the ith entity of
    /// color 'col'.
    std::map<const std::vector<uint>,
             std::pair<MeshFunction<uint>, std::vector<std::vector<uint> > > > coloring;

  private:

    // True if a facet is an exterior facet, false otherwise
    boost::scoped_ptr<MeshFunction<bool> >_exterior_facet;

  };

}

#endif
