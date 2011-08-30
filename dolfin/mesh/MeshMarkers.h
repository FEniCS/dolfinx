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
// Last changed: 2011-08-30

#ifndef __MESH_MARKERS_H
#define __MESH_MARKERS_H

#include <vector>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/Variable.h>

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

    /// Create empty mesh markers on given mesh
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to create mesh markers on.
    MeshMarkers(const Mesh& mesh);

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
    boost::shared_ptr<Mesh> _mesh;

    // The markers
    std::vector<std::vector<T> > _markers;

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
