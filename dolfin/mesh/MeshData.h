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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
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

  // FIXME: Remove space with underscore in names

  /// The class MeshData is a container for auxiliary mesh data,
  /// represented either as MeshFunctions over topological mesh
  /// entities, arrays or maps. Each dataset is identified by a unique
  /// user-specified string. Only uint-valued data are currently
  /// supported.
  ///
  /// The following named mesh data are recognized by DOLFIN:
  ///
  /// Boundary indicators
  ///
  ///   "boundary_facet_cells"   - Array<uint> of size num_facets
  ///   "boundary_facet_numbers" - Array<uint> of size num_facets
  ///   "boundary_indicators"    - Array<uint> of size num_facets
  ///   "material_indicators"    - MeshFunction<uint> of dimension D
  ///
  /// Subdomain indicators
  ///
  ///   "cell_domains"           - MeshFunction<uint> of dimension D
  ///   "interior_facet_domains" - MeshFunction<uint> of dimension D - 1
  ///   "exterior_facet_domains" - MeshFunction<uint> of dimension D - 1
  ///
  /// Facet orientation (used for assembly over interior facets)
  ///
  ///   "facet orientation" - MeshFunction<uint> of dimension D - 1
  ///
  /// Boundary extraction
  ///
  ///   "vertex map" - MeshFunction<uint> of dimension 0
  ///   "cell map"   - MeshFunction<uint> of dimension D
  ///
  /// Mesh partitioning
  ///
  ///   (moved to ParallelData) "global entity indices %d" - MeshFunction<uint> of dimension 0, 1, ..., D
  ///   (moved to ParallelData) "exterior facets"          - MeshFunction<uint> of dimension D - 1
  ///   (moved to ParallelData) "num global entities"      - Array<uint> of size D + 1
  ///   (moved to ParallelData) "overlap"                  - vector mapping
  ///
  /// Sub meshes
  ///
  ///   "global vertex indices" - MeshFunction<uint> of dimension 0
  ///
  /// Mesh coloring
  ///
  ///   "colors-%D-%d-%1"   - MeshFunction<uint> of dimension D with colors based on connectivity %d
  ///   "num colored cells" - Array<uint> listing the number of cells of each color
  ///   "colored cells %d"  - Array<uint> of cell indices with colors 0, 1, 2, ...

  class MeshData : public Variable
  {
  public:

    /// Constructor
    MeshData(Mesh& mesh);

    /// Destructor
    ~MeshData();

    /// Assignment operator
    const MeshData& operator= (const MeshData& data);

    /// Clear all data
    void clear();

    //--- Creation of data ---

    /// Create MeshFunction with given name (uninitialized)
    boost::shared_ptr<MeshFunction<unsigned int> > create_mesh_function(std::string name);

    /// Create MeshFunction with given name and dimension
    boost::shared_ptr<MeshFunction<unsigned int> > create_mesh_function(std::string name, uint dim);

    /// Create empty array (vector) with given name
    std::vector<uint>* create_array(std::string name);

    /// Create array (vector) with given name and size
    std::vector<uint>* create_array(std::string name, uint size);

    /// Create mapping from uint to uint with given name
    std::map<uint, uint>* create_mapping(std::string name);

    /// Create mapping from uint to vector of uint with given name
    std::map<uint, std::vector<uint> >* create_vector_mapping(std::string name);

    //--- Retrieval of data ---

    /// Return MeshFunction with given name (returning zero if data is not available)
    boost::shared_ptr<MeshFunction<unsigned int> > mesh_function(const std::string name) const;

    /// Return array with given name (returning zero if data is not available)
    std::vector<uint>* array(const std::string name) const;

    /// Return array with given name postfixed by " %d" (returning zero if data is not available)
    std::vector<uint>* array(const std::string name, uint number) const;

    /// Return mapping with given name (returning zero if data is not available)
    std::map<uint, uint>* mapping(const std::string name) const;

    /// Return vector mapping with given name (returning zero if data is not available)
    std::map<uint, std::vector<uint> >* vector_mapping(const std::string name) const;

    //--- Removal of data ---

    /// Erase MeshFunction with given name
    void erase_mesh_function(const std::string name);

    /// Erase array with given name
    void erase_array(const std::string name);

    /// Erase mapping with given name
    void erase_mapping(const std::string name);

    /// Erase vector mapping with given name
    void erase_vector_mapping(const std::string name);

    //--- Misc ---

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Friends
    friend class XMLFile;
    friend class XMLMeshData;

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
