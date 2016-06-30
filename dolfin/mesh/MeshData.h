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
// Last changed: 2011-09-15

#ifndef __MESH_DATA_H
#define __MESH_DATA_H

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  class Mesh;

  /// The class MeshData is a container for auxiliary mesh data,
  /// represented either as arrays or maps. Each dataset is identified
  /// by a unique user-specified string. Only std::size_t-valued data
  /// are currently supported.
  ///
  /// Auxiliary mesh data may be attached to a mesh by users as a
  /// convenient way to store data associated with a mesh. It is also
  /// used internally by DOLFIN to communicate data associated with
  /// meshes. The following named mesh data are recognized by DOLFIN:
  ///
  /// Facet orientation (used for assembly over interior facets)
  ///
  ///   * "facet_orientation"  - _std:vector_ <std::size_t> of dimension D - 1
  ///
  /// Sub meshes (used by the class SubMesh)
  ///
  ///   * "parent_vertex_indices" - _std::vector_ <std::size_t> of dimension 0
  ///
  /// Note to developers: use underscore in names in place of spaces.

  class MeshData : public Variable
  {
  public:

    /// Constructor
    MeshData();

    /// Destructor
    ~MeshData();

    /// Assignment operator
    ///
    /// @param     data (_MeshData_)
    ///         Another MeshData object.
    const MeshData& operator= (const MeshData& data);

    /// Clear all data
    void clear();

    //--- Query of data ---

    /// Check is array exists
    ///
    /// @param     name (std::string)
    ///         The name of the array.
    ///
    /// @return     bool
    ///         True is array exists, false otherwise.
    bool exists(std::string name, std::size_t dim) const;

    //--- Creation of data ---

    /// Create array (vector) with given name and size
    ///
    /// @param     name (std::string)
    ///         The name of the array.
    /// @param     size (std::size_t)
    ///         The size (length) of the array.
    ///
    ///  @return    std::vector<std::size_t>
    ///         The array.
    std::vector<std::size_t>& create_array(std::string name, std::size_t dim);

    //--- Retrieval of data ---

    /// Return array with given name (returning zero if data is not
    /// available)
    ///
    /// @param     name (std::string)
    ///         The name of the array.
    ///  @param dim (std::size_t)
    ///          Dimension.
    /// @return    std::vector<std::size_t>
    ///         The array.
    std::vector<std::size_t>& array(std::string name, std::size_t dim);

    /// Return array with given name (returning zero if data is not
    /// available)
    ///
    /// @param     name (std::string)
    ///         The name of the array.
    ///
    /// @return      std::vector<std::size_t>
    ///         The array.
    const std::vector<std::size_t>& array(std::string name,
                                          std::size_t dim) const;

    //--- Removal of data ---

    /// Erase array with given name
    ///
    /// @param     name (std::string)
    ///         The name of the array.
    void erase_array(const std::string name, std::size_t dim);

    //--- Misc ---

    /// Return informal string representation (pretty-print)
    ///
    /// @param     verbose (bool)
    ///         Flag to turn on additional output.
    ///
    /// @return     std::string
    ///         An informal representation.
    std::string str(bool verbose) const;

    /// Friends
    friend class XMLMesh;

  private:

    // Check if name is deprecated
    void check_deprecated(std::string name) const;

    // A map from named mesh array data to vector for dim
    std::vector<std::map<std::string, std::vector<std::size_t> > > _arrays;

    // List of deprecated named data
    std::vector<std::string> _deprecated_names;

  };

}

#endif
