// Copyright (C) 2008-2009 Anders Logg
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

  // FIXME: Replace space with underscore in names

  /// The class MeshData is a container for auxiliary mesh data,
  /// represented either as _MeshFunction_ over topological mesh
  /// entities, arrays or maps. Each dataset is identified by a unique
  /// user-specified string. Only uint-valued data are currently
  /// supported.
  ///
  /// The following named mesh data are recognized by DOLFIN:
  ///
  /// Boundary indicators
  ///
  ///   * "boundary_facet_cells"   -  _Array_ <uint> of size num_facets
  ///   * "boundary_facet_numbers" -  _Array_ <uint> of size num_facets
  ///   * "boundary_indicators"    -  _Array_ <uint> of size num_facets
  ///   * "material_indicators"    -  _MeshFunction_ <uint> of dimension D
  ///
  ///
  /// Subdomain indicators
  ///
  ///   * "cell_domains"           - _MeshFunction_ <uint> of dimension D
  ///   * "interior_facet_domains" - _MeshFunction_ <uint> of dimension D - 1
  ///   * "exterior_facet_domains" - _MeshFunction_ <uint> of dimension D - 1
  ///
  ///
  /// Facet orientation (used for assembly over interior facets)
  ///
  ///   * "facet orientation" - _MeshFunction_ <uint> of dimension D - 1
  ///
  ///
  /// Boundary extraction
  ///
  ///   * (removed, is now a member function of BoundaryMesh) "vertex map" - _MeshFunction_ <uint> of dimension 0
  ///   * (removed, is now a member function of BoundaryMesh) "cell map"   - _MeshFunction_ <uint> of dimension D
  ///
  ///
  /// Mesh partitioning
  ///
  ///   * (moved to ParallelData) "global entity indices %d" - _MeshFunction_ <uint> of dimension 0, 1, ..., D
  ///   * (moved to ParallelData) "exterior facets"          - _MeshFunction_ <uint> of dimension D - 1
  ///   * (moved to ParallelData) "num global entities"      - _Array_ <uint> of size D + 1
  ///   * (moved to ParallelData) "overlap"                  - vector mapping
  ///
  ///
  /// Sub meshes
  ///
  ///   * "global vertex indices" - _MeshFunction_ <uint> of dimension 0
  ///
  ///
  /// Mesh
  ///
  ///   * "colors-%D-%d-%1"   - _MeshFunction_ <uint> of dimension D with colors based on connectivity %d
  ///   * "num colored cells" - _Array_ <uint> listing the number of cells of each color
  ///   * "colored cells %d"  - _Array_ <uint> of cell indices with colors 0, 1, 2, ...
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
    boost::shared_ptr<MeshFunction<unsigned int> > create_mesh_function(std::string name);

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
    boost::shared_ptr<MeshFunction<unsigned int> > create_mesh_function(std::string name, uint dim);

    /// Create empty array (vector) with given name
    ///
    /// *Arguments*
    ///     name (std::string)
    ///         The name of the array.
    ///
    /// *Returns*
    ///     std::vector<uint>
    ///         The array.
    std::vector<uint>* create_array(std::string name);

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
    std::vector<uint>* create_array(std::string name, uint size);

    /// Create mapping from uint to uint with given name
    ///
    /// *Arguments*
    ///     name (std::string)
    ///         The name of the map.
    ///
    /// *Returns*
    ///     std::map<uint, uint>
    ///         The map.
    std::map<uint, uint>* create_mapping(std::string name);

    /// Create mapping from uint to vector of uint with given name
    ///
    /// *Arguments*
    ///     name (std::string)
    ///         The name of the map.
    ///
    /// *Returns*
    ///     std::map<uint, std::vector<uint> >
    ///         The map.
    std::map<uint, std::vector<uint> >* create_vector_mapping(std::string name);

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
    std::vector<uint>* array(const std::string name) const;

    /// Return array with given name postfixed by " %d" (returning zero
    /// if data is not available)
    ///
    /// *Arguments*
    ///     name (std::string)
    ///         The name.
    ///     number (uint)
    ///         The number.
    ///
    /// *Returns*
    ///     std::vector<uint>
    ///         The array.
    std::vector<uint>* array(const std::string name, uint number) const;

    /// Return mapping with given name (returning zero if data is not
    /// available)
    ///
    /// *Arguments*
    ///     name (std::string)
    ///         The name of the map.
    ///
    /// *Returns*
    ///     std::map<uint, uint>
    ///         The map.
    std::map<uint, uint>* mapping(const std::string name) const;

    /// Return vector mapping with given name (returning zero if data
    /// is not available)
    ///
    /// *Arguments*
    ///     name (std::string)
    ///         The name of the map
    ///
    /// *Returns*
    ///     std::map<uint, std::vector<uint> >
    ///         The vector mapping.
    std::map<uint, std::vector<uint> >* vector_mapping(const std::string name) const;

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

    /// Erase mapping with given name
    ///
    /// *Arguments*
    ///     name (std::string)
    ///         The name of the mapping.
    void erase_mapping(const std::string name);

    /// Erase vector mapping with given name
    ///
    /// *Arguments*
    ///     name (std::string)
    ///         The name of the vector mapping.
    void erase_vector_mapping(const std::string name);

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
    std::map<std::string, std::vector<uint>* > arrays;

    // A map from named mesh data to mapping
    std::map<std::string, std::map<uint, uint>* > mappings;

    // A map from named mesh data to vector mapping
    std::map<std::string, std::map<uint, std::vector<uint> >* > vector_mappings;

  };

}

#endif
