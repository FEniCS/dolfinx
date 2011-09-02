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
// Last changed: 2011-09-02

#ifndef __MESH_MARKERS_H
#define __MESH_MARKERS_H

#include <vector>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/Variable.h>
#include "MeshEntity.h"
#include "Cell.h"
#include "MeshFunction.h"

namespace dolfin
{

  /// The MeshMarkers class can be used to store data associated with
  /// a subset of the entities of a mesh of a given topological
  /// dimension. It differs from the MeshFunction class in two ways.
  /// First, data does not need to be associated with all entities
  /// (only a subset). Second, data is associated with entities
  /// through the corresponding cell index and local entity number
  /// (relative to the cell), not by global entity index, which means
  /// that data may be stored robustly to file.

  template <class T> class MeshMarkers : public Variable
  {
  public:

    /// Create empty mesh markers
    MeshMarkers();

    /// Create empty mesh markers of given dimension on given mesh
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to create mesh markers on.
    ///     dim (uint)
    ///         The mesh entity dimension for the mesh markers.
    MeshMarkers(const Mesh& mesh, uint dim);

    /// Create empty mesh markers of given dimension on given mesh
    /// (shared pointer version)
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to create mesh markers on.
    ///     dim (uint)
    ///         The mesh entity dimension for the mesh markers.
    MeshMarkers(boost::shared_ptr<const Mesh> mesh, uint dim);

    /// Destructor
    ~MeshMarkers()
    {}

    /// Return mesh associated with mesh markers
    ///
    /// *Returns*
    ///     _Mesh_
    ///         The mesh.
    const Mesh& mesh() const;

    /// Return mesh associated with mesh markers (shared pointer version)
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

    /// Set marker value for given entity index
    ///
    /// *Arguments*
    ///     entity_index (uint)
    ///         Index of the entity.
    ///     marker_value (T).
    ///         The value of the marker.
    void set_value(uint entity_index, T& marker_value);

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

    // The markers
    std::vector<std::pair<std::pair<uint, uint>, T> > _markers;

    /// Topological dimension
    uint _dim;

  };

  //---------------------------------------------------------------------------
  // Implementation of MeshMarkers
  //---------------------------------------------------------------------------
  template <class T>
  MeshMarkers<T>::MeshMarkers()
    : Variable("m", "unnamed MeshMarkers"), _dim(0)
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <class T>
  MeshMarkers<T>::MeshMarkers(const Mesh& mesh, uint dim)
    : Variable("m", "unnamed MeshMarkers"),
      _mesh(reference_to_no_delete_pointer(mesh)), _dim(dim)
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <class T>
  MeshMarkers<T>::MeshMarkers(boost::shared_ptr<const Mesh> mesh, uint dim)
    : Variable("m", "unnamed MeshMarkers"),
      _mesh(mesh), _dim(dim)
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <class T>
  const Mesh& MeshMarkers<T>::mesh() const
  {
    assert(_mesh);
    return *_mesh;
  }
  //---------------------------------------------------------------------------
  template <class T>
  boost::shared_ptr<const Mesh> MeshMarkers<T>::mesh_ptr() const
  {
    return _mesh;
  }
  //---------------------------------------------------------------------------
  template <class T>
  uint MeshMarkers<T>::dim() const
  {
    return _dim;
  }
  //---------------------------------------------------------------------------
  template <class T>
  uint MeshMarkers<T>::size() const
  {
    return _markers.size();
  }
  //---------------------------------------------------------------------------
  template <class T>
  void MeshMarkers<T>::set_value(uint entity_index, T& marker_value)
  {
    assert(_mesh);

    // Get mesh connectivity d --> D
    const uint D = _mesh->topology().dim();
    const MeshConnectivity& connectivity = _mesh->topology()(_dim, D);

    // Find the cell
    assert(connectivity.size(entity_index) > 0);
    MeshEntity entity(*_mesh, _dim, entity_index);
    Cell cell(*_mesh, connectivity(entity_index)[0]); // choose first

    // Find the local entity index
    const uint local_entity = cell.index(entity);

    // Add marker
    std::pair<uint, uint> pos(std::make_pair(cell.index(), local_entity));
    _markers.push_back(std::make_pair(pos, marker_value));
  }
  //---------------------------------------------------------------------------
  template <class T>
  void MeshMarkers<T>::extract_mesh_function(MeshFunction<T>& mesh_function) const
  {
    assert(_mesh);

    // Initialize mesh function
    mesh_function.init(_mesh, _dim);

    // Get mesh connectivity D --> d
    const uint D = _mesh->topology().dim();
    const MeshConnectivity& connectivity = _mesh->topology()(D, _dim);

    // Get maximum value of marker. Note that this requires that the
    // type T can be intialized to zero, supports std::max and can
    // be incremented by 1.
    T maxval = T(0);
    for (uint i = 0; i < _markers.size(); i++)
      maxval = std::max(maxval, _markers[i][2]);

    // Set all value of mesh function to maximum value (not all will
    // be set) by markers below
    mesh_function.set_all(maxval);

    // Iterate over all markers
    for (uint i = 0; i < _markers.size(); i++)
    {
      // Get marker data
      const std::vector<uint>& marker = _markers[i];
      const uint cell_index   = marker[0];
      const uint local_entity = marker[1];
      const T marker_value    = marker[2];

      // Get global entity index
      const uint entity_index = connectivity(cell_index)[local_entity];

      // Set boundary indicator for facet
      mesh_function[entity_index] = marker_value;
    }
  }
  //---------------------------------------------------------------------------
  template <class T>
  std::string MeshMarkers<T>::str(bool verbose) const
  {
    std::stringstream s;

    if (verbose)
    {
      s << str(false) << std::endl << std::endl;
      warning("Verbose output of MeshMarkers must be implemented manually.");
    }
    else
      s << "<MeshMarkers of topological dimension " << dim()
        << " containing " << size() << " values>";

    return s.str();
  }
  //---------------------------------------------------------------------------

}

#endif
