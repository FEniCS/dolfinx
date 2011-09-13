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
// First added:  2006-08-30
// Last changed: 2011-09-13

#ifndef __MESH_VALUE_COLLECTION_H
#define __MESH_VALUE_COLLECTION_H

#include <map>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/Variable.h>
#include "MeshEntity.h"
#include "Cell.h"
#include "MeshFunction.h"

namespace dolfin
{

  /// The MeshValueCollection class can be used to store data
  /// associated with a subset of the entities of a mesh of a given
  /// topological dimension. It differs from the MeshFunction class in
  /// two ways. First, data does not need to be associated with all
  /// entities (only a subset). Second, data is associated with
  /// entities through the corresponding cell index and local entity
  /// number (relative to the cell), not by global entity index, which
  /// means that data may be stored robustly to file.

  template <class T> class MeshValueCollection : public Variable
  {
  public:

    /// Create empty mesh value collection of given dimension on given mesh
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to create mesh value collection on.
    ///     dim (uint)
    ///         The mesh entity dimension for the mesh value collection.
    MeshValueCollection(const Mesh& mesh, uint dim);

    /// Create empty mesh value collection of given dimension on given mesh
    /// (shared pointer version)
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to create mesh value collection on.
    ///     dim (uint)
    ///         The mesh entity dimension for the mesh value collection.
    MeshValueCollection(boost::shared_ptr<const Mesh> mesh, uint dim);

    /// Destructor
    ~MeshValueCollection()
    {}

    /// Return mesh associated with mesh value collection
    ///
    /// *Returns*
    ///     _Mesh_
    ///         The mesh.
    const Mesh& mesh() const;

    /// Return mesh associated with mesh value collection (shared pointer version)
    ///
    /// *Returns*
    ///     _Mesh_
    ///         The mesh.
    boost::shared_ptr<const Mesh> mesh_ptr() const;

    /// Return topological dimension
    ///
    /// *Returns*
    ///     uint
    ///         The dimension.
    uint dim() const;

    /// Return size (number of entities in subset)
    ///
    /// *Returns*
    ///     uint
    ///         The size.
    uint size() const;

    /// Set marker value for given entity defined by a cell index and
    /// a local entity index
    ///
    /// *Arguments*
    ///     cell_index (uint)
    ///         The index of the cell.
    ///     local_entity (uint)
    ///         The local index of the entity relative to the cell.
    ///     marker_value (T)
    ///         The value of the marker.
    void set_value(uint cell_index, uint local_entity, const T& marker_value);

    /// Set value for given entity index
    ///
    /// *Arguments*
    ///     entity_index (uint)
    ///         Index of the entity.
    ///     value (T).
    ///         The value.
    void set_value(uint entity_index, const T& value);

    /// Get all values
    ///
    /// *Returns*
    ///     std::map<std::pair<uint, uint>, T>
    ///         A map from positions to values.
    std::map<std::pair<uint, uint>, T>& values();

    /// Get all values (const version)
    ///
    /// *Returns*
    ///     std::map<std::pair<uint, uint>, T>
    ///         A map from positions to values.
    const std::map<std::pair<uint, uint>, T>& values() const;

    /// Clear all values
    void clear();

    /// Extract data for corresponding MeshFunction
    ///
    /// *Arguments*
    ///     mesh_function (_MeshFunction_)
    ///         The MeshFunction to be computed.
    void extract_mesh_function(MeshFunction<T>& mesh_function) const;

    /// Return informal string representation (pretty-print)
    ///
    /// *Arguments*
    ///     verbose (bool)
    ///         Flag to turn on additional output.
    ///
    /// *Returns*
    ///     std::string
    ///         An informal representation.
    std::string str(bool verbose) const;

  private:

    // The mesh
    boost::shared_ptr<const Mesh> _mesh;

    // The values
    std::map<std::pair<uint, uint>, T> _values;

    /// Topological dimension
    uint _dim;

  };

  //---------------------------------------------------------------------------
  // Implementation of MeshValueCollection
  //---------------------------------------------------------------------------
  template <class T>
  MeshValueCollection<T>::MeshValueCollection(const Mesh& mesh, uint dim)
    : Variable("m", "unnamed MeshValueCollection"),
      _mesh(reference_to_no_delete_pointer(mesh)), _dim(dim)
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <class T>
  MeshValueCollection<T>::MeshValueCollection(boost::shared_ptr<const Mesh> mesh, uint dim)
    : Variable("m", "unnamed MeshValueCollection"),
      _mesh(mesh), _dim(dim)
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <class T>
  const Mesh& MeshValueCollection<T>::mesh() const
  {
    assert(_mesh);
    return *_mesh;
  }
  //---------------------------------------------------------------------------
  template <class T>
  boost::shared_ptr<const Mesh> MeshValueCollection<T>::mesh_ptr() const
  {
    return _mesh;
  }
  //---------------------------------------------------------------------------
  template <class T>
  uint MeshValueCollection<T>::dim() const
  {
    return _dim;
  }
  //---------------------------------------------------------------------------
  template <class T>
  uint MeshValueCollection<T>::size() const
  {
    return _values.size();
  }
  //---------------------------------------------------------------------------
  template <class T>
  void MeshValueCollection<T>::set_value(uint cell_index,
                                         uint local_entity,
                                         const T& value)
  {
    std::pair<uint, uint> pos(std::make_pair(cell_index, local_entity));
    _values[pos] = value;
  }
  //---------------------------------------------------------------------------
  template <class T>
  void MeshValueCollection<T>::set_value(uint entity_index, const T& value)
  {
    assert(_mesh);

    // Get mesh connectivity d --> D
    const uint D = _mesh->topology().dim();
    const MeshConnectivity& connectivity = _mesh->topology()(_dim, D);

    // Find the cell
    // FIXME: Make this an if statement? It crashes if not initialized.
    assert(connectivity.size(entity_index) > 0);
    MeshEntity entity(*_mesh, _dim, entity_index);
    Cell cell(*_mesh, connectivity(entity_index)[0]); // choose first

    // Find the local entity index
    const uint local_entity = cell.index(entity);

    // Add value
    std::pair<uint, uint> pos(std::make_pair(cell.index(), local_entity));
    _values[pos] = value;
  }
  //---------------------------------------------------------------------------
  template <class T>
  std::map<std::pair<uint, uint>, T>& MeshValueCollection<T>::values()
  {
    return _values;
  }
  //---------------------------------------------------------------------------
  template <class T>
  const std::map<std::pair<uint, uint>, T>& MeshValueCollection<T>::values() const
  {
    return _values;
  }
  //---------------------------------------------------------------------------
  template <class T>
  void MeshValueCollection<T>::clear()
  {
    _values.clear();
  }
  //---------------------------------------------------------------------------
  template <class T>
  void MeshValueCollection<T>::extract_mesh_function(MeshFunction<T>& mesh_function) const
  {
    assert(_mesh);

    // Initialize mesh function
    mesh_function.init(*_mesh, _dim);

    // Get mesh connectivity D --> d
    const uint D = _mesh->topology().dim();
    const MeshConnectivity& connectivity = _mesh->topology()(D, _dim);

    // Get maximum value. Note that this requires that the type T can
    // be intialized to zero, supports std::max and can be incremented
    // by 1.
    T maxval = T(0);
    typename std::map<std::pair<uint, uint>, T>::const_iterator it;
    for (it = _values.begin(); it != _values.end(); ++it)
      maxval = std::max(maxval, it->second);

    // Set all value of mesh function to maximum value (not all will
    // be set) by markers below
    mesh_function.set_all(maxval + 1);

    // Iterate over all values
    for (it = _values.begin(); it != _values.end(); ++it)
    {
      // Get marker data
      const uint cell_index   = it->first.first;
      const uint local_entity = it->first.second;
      const T value    = it->second;

      // Get global entity index
      const uint entity_index = connectivity(cell_index)[local_entity];

      // Set boundary indicator for facet
      mesh_function[entity_index] = value;
    }
  }
  //---------------------------------------------------------------------------
  template <class T>
  std::string MeshValueCollection<T>::str(bool verbose) const
  {
    std::stringstream s;

    if (verbose)
    {
      s << str(false) << std::endl << std::endl;
      warning("Verbose output of MeshValueCollection must be implemented manually.");
    }
    else
      s << "<MeshValueCollection of topological dimension " << dim()
        << " containing " << size() << " values>";

    return s.str();
  }
  //---------------------------------------------------------------------------

}

#endif
