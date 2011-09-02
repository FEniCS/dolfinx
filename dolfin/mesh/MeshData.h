// Copyright (C) 2008-2011 Anders Logg
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
// Modified by Niclas Jansson, 2008.
// Modified by Garth N. Wells, 2011.
//
// First added:  2008-05-19
// Last changed: 2011-03-10

#ifndef __MESH_DATA_H
#define __MESH_DATA_H

#include <map>
#include <string>
#include <utility>
#include <vector>
#include <boost/shared_ptr.hpp>

#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  class Mesh;
  template <class T> class MeshFunction;

  /// The class MeshData is a container for auxiliary mesh data,
  /// represented either as _MeshFunction_ over topological mesh
  /// entities, arrays or maps. Each dataset is identified by a unique
  /// user-specified string. Only uint-valued data are currently
  /// supported.
  ///
  /// Auxiliary mesh data may be attached to a mesh by users as a
  /// convenient way to store data associated with a mesh. It is also
  /// used internally by DOLFIN to communicate data associated with
  /// meshes. The following named mesh data are recognized by DOLFIN:
  ///
  /// Boundary indicators
  ///
  ///   * "boundary_facet_cells"   -  _Array_ <uint> of size num_facets
  ///   * "boundary_facet_numbers" -  _Array_ <uint> of size num_facets
  ///   * "boundary_indicators"    -  _Array_ <uint> of size num_facets
  ///   * "material_indicators"    -  _MeshFunction_ <uint> of dimension D
  ///
  /// Subdomain indicators
  ///
  ///   * "cell_domains"           - _MeshFunction_ <uint> of dimension D
  ///   * "interior_facet_domains" - _MeshFunction_ <uint> of dimension D - 1
  ///   * "exterior_facet_domains" - _MeshFunction_ <uint> of dimension D - 1
  ///
  /// Facet orientation (used for assembly over interior facets)
  ///
  ///   * "facet_orientation"      - _MeshFunction_ <uint> of dimension D - 1
  ///
  /// Sub meshes (used by the class SubMesh)
  ///
  ///   * "parent_vertex_indices"  - _MeshFunction_ <uint> of dimension 0
  ///
  /// Note to developers: use underscore in names in place of spaces.

  class MeshData : public Variable
  {
  public:

    /// Constructor
    MeshData(Mesh& mesh);

    /// Destructor
    ~MeshData();

    /// Assignment operator
    ///
    /// *Arguments*
    ///     data (_MeshData_)
    ///         Another MeshData object.
    const MeshData& operator= (const MeshData& data);

    /// Clear all data
    void clear();

    //--- Creation of data ---

    /// Create MeshFunction with given name (uninitialized)
    ///
    /// *Arguments*
    ///     name (std::string)
    ///         The name of the mesh function.
    ///
    /// *Returns*
    ///     _MeshFunction_ <unsigned int>
    ///         The mesh function.
    boost::shared_ptr<MeshFunction<unsigned int> >
    create_mesh_function(std::string name);

    /// Create MeshFunction with given name and dimension
    ///
    /// *Arguments*
    ///     name (std::string)
    ///         The name of the mesh function.
    ///     dim (uint)
    ///         The dimension of the mesh function.
    ///
    /// *Returns*
    ///     _MeshFunction_ <unsigned int>
    ///         The mesh function.
    boost::shared_ptr<MeshFunction<unsigned int> >
    create_mesh_function(std::string name, uint dim);

    /// Create empty array (vector) with given name
    ///
    /// *Arguments*
    ///     name (std::string)
    ///         The name of the array.
    ///
    /// *Returns*
    ///     std::vector<uint>
    ///         The array.
    boost::shared_ptr<std::vector<uint> > create_array(std::string name);

    /// Create array (vector) with given name and size
    ///
    /// *Arguments*
    ///     name (std::string)
    ///         The name of the array.
    ///     size (unit)
    ///         The size (length) of the array.
    ///
    /// *Returns*
    ///     std::vector<uint>
    ///         The array.
    boost::shared_ptr<std::vector<uint> > create_array(std::string name, uint size);

    //--- Retrieval of data ---

    /// Return MeshFunction with given name (returning zero if data is
    /// not available)
    ///
    /// *Arguments*
    ///     name (std::string)
    ///         The name of the MeshFunction.
    ///
    /// *Returns*
    ///     _MeshFunction_ <unsigned int>
    ///         The mesh function with given name
    boost::shared_ptr<MeshFunction<unsigned int> > mesh_function(const std::string name) const;

    /// Return array with given name (returning zero if data is not
    /// available)
    ///
    /// *Arguments*
    ///     name (std::string)
    ///         The name of the array.
    ///
    /// *Returns*
    ///     std::vector<uint>
    ///         The array.
    boost::shared_ptr<std::vector<uint> > array(const std::string name) const;

    //--- Removal of data ---

    /// Erase MeshFunction with given name
    ///
    /// *Arguments*
    ///     name (std::string)
    ///         The name of the mesh function
    void erase_mesh_function(const std::string name);

    /// Erase array with given name
    ///
    /// *Arguments*
    ///     name (std::string)
    ///         The name of the array.
    void erase_array(const std::string name);

    //--- Misc ---

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

    /// Friends
    friend class XMLMesh;
    friend class OldXMLMeshData;
    friend class MeshPartitioning;

  private:

    // The mesh
    Mesh& mesh;

    // A map from named mesh data to MeshFunctions
    std::map<std::string, boost::shared_ptr<MeshFunction<uint> > > mesh_functions;

    // A map from named mesh data to vector
    std::map<std::string, boost::shared_ptr<std::vector<uint> > > arrays;

  };

}

#endif
