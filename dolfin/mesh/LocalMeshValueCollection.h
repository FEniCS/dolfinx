// Copyright (C) 201 Garth N. Wells
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
// Modified by Anders Logg, 2008-2009.
//
// First added:  2008-11-28
// Last changed: 2011-03-25
//
// Modified by Anders Logg, 2008-2009.
// Modified by Kent-Andre Mardal, 2011.


#ifndef __LOCAL_MESH_VALUE_COLLECTION_H
#define __LOCAL_MESH_VALUE_COLLECTION_H

#include <map>
#include <utility>
#include <vector>
#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include <dolfin/log/log.h>

namespace dolfin
{

  template <typename T> class MeshValueCollection;

  /// This class stores mesh data on a local processor corresponding
  /// to a portion of a MeshValueCollection.

  template <typename T>
  class LocalMeshValueCollection
  {
  public:

    /// Create local mesh data for given LocalMeshValueCollection
    LocalMeshValueCollection(const MeshValueCollection<T>& values, uint dim);

    /// Destructor
    ~LocalMeshValueCollection() {}

    /// Return dimension of cell entity
    uint dim () const
    { return _dim; }

    /// Return data
    const std::vector<std::pair<std::pair<uint, uint>, T> >& values() const
    { return _values; }

  private:

    /// Topological dimension
    const uint _dim;

    // MeshValueCollection values (cell_index, local_index), value))
    std::vector<std::pair<std::pair<uint, uint>, T> >  _values;

  };

  //---------------------------------------------------------------------------
  // Implementation of LocalMeshValueCollection
  //---------------------------------------------------------------------------
  template <typename T>
  LocalMeshValueCollection<T>::LocalMeshValueCollection(const MeshValueCollection<T>& values,
                                                        dolfin::uint dim)
      : _dim(dim)
  {
    // Prepare data
    std::vector<std::vector<uint> > indices;
    std::vector<std::vector<T> > v;

    // Extract data on main process and split among processes
    if (MPI::is_broadcaster())
    {
      // Get number of processes
      const uint num_processes = MPI::num_processes();
      indices.resize(num_processes);
      v.resize(num_processes);

      const std::map<std::pair<uint, uint>, T>& vals = values.values();
      for (uint p = 0; p < num_processes; p++)
      {
        const std::pair<uint, uint> local_range = MPI::local_range(p, vals.size());
        typename std::map<std::pair<uint, uint>, T>::const_iterator it = vals.begin();
        std::advance(it, local_range.first);
        for (uint i = local_range.first; i < local_range.second; ++i)
        {
          indices[p].push_back(it->first.first);
          indices[p].push_back(it->first.second);
          v[p].push_back(it->second);
          std::advance(it, 1);
        }
      }
    }

    // Scatter data
    MPI::scatter(indices);
    MPI::scatter(v);
    assert(2*v[0].size() == indices[0].size());

    // Unpack
    for (uint i = 0; i < v[0].size(); ++i)
    {
      const uint cell_index = indices[0][2*i];
      const uint local_entity_index = indices[0][2*i + 1];
      const T value = v[0][i];

      _values.push_back( std::make_pair( std::make_pair(cell_index, local_entity_index) , value) );
    }
  }
  //---------------------------------------------------------------------------

}

#endif
