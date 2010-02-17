// Copyright (C) 2008-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson, 2008.
//
// First added:  2008-05-19
// Last changed: 2010-02-17

#ifndef __MESH_DATA_H
#define __MESH_DATA_H

#include <map>
#include <dolfin/common/Variable.h>
#include <dolfin/common/types.h>

namespace dolfin
{

  class Mesh;
  template <class T> class MeshFunction;

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
  ///   "exterior facet domains" - MeshFunction<uint> of dimension D - 1
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
  ///   "global entity indices %d" - MeshFunction<uint> of dimension 0, 1, ..., D
  ///   "exterior facets"          - MeshFunction<uint> of dimension D - 1
  ///   "num global entities"      - Array<uint> of size D + 1
  ///   "overlap"                  - vector mapping
  ///
  /// Sub meshes
  ///
  ///    "global vertex indices" - MeshFunction<uint> of dimension 0
  ///
  /// Mesh refinement
  ///
  ///   "boundary facet cells"   - MeshFunction<uint> of dimension 0, 1, ..., D
  ///   "boundary facet numbers" - MeshFunction<uint> of dimension 0, 1, ..., D
  ///   "boundary indicators"    - MeshFunction<uint> of dimension 0, 1, ..., D
  ///   "material indicators"    - MeshFunction<uint> of dimension D

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
    MeshFunction<uint>* create_mesh_function(std::string name);

    /// Create MeshFunction with given name and dimension
    MeshFunction<uint>* create_mesh_function(std::string name, uint dim);

    /// Create array (vector) with given name and size
    std::vector<uint>* create_array(std::string name, uint size);

    /// Create mapping from uint to uint with given name
    std::map<uint, uint>* create_mapping(std::string name);

    /// Create mapping from uint to vector of uint with given name
    std::map<uint, std::vector<uint> >* create_vector_mapping(std::string name);

    //--- Retrieval of data ---

    /// Return MeshFunction with given name (returning zero if data is not available)
    MeshFunction<uint>* mesh_function(const std::string name) const;

    /// Return array with given name (returning zero if data is not available)
    std::vector<uint>* array(const std::string name) const;

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
    std::map<std::string, MeshFunction<uint>* > mesh_functions;

    // A map from named mesh data to vector
    std::map<std::string, std::vector<uint>* > arrays;

    // A map from named mesh data to mapping
    std::map<std::string, std::map<uint, uint>* > mappings;

    // A map from named mesh data to vector mapping
    std::map<std::string, std::map<uint, std::vector<uint> >* > vector_mappings;

  };

}

#endif
