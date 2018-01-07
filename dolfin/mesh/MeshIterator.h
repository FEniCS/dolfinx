#pragma once

#include <iterator>

#include "Mesh.h"
#include "MeshConnectivity.h"
#include "MeshEntity.h"
#include "MeshTopology.h"

namespace dolfin
{

  /// Iterator for iterating over entities of a Mesh
  // FIXME: Add 'MeshIterator Mesh::iterator(std::size_t dim);'?
  class MeshIterator : public std::iterator<std::forward_iterator_tag, MeshEntity>
  {
  public:

    /// Constructor from Mesh and entity dimension
    MeshIterator(const Mesh& mesh, std::size_t dim, std::size_t pos)
      : _entity(mesh, dim, pos)
    {
      // Do nothing
    }

    /// Copy constructor
    MeshIterator(const MeshIterator& it) : _entity(it._entity)
    {
      // Do nothing
    }

    /// Copy assignment
    const MeshIterator& operator= (const MeshIterator& m)
    {
      _entity = m._entity;
      return *this;
    }

    MeshIterator& operator++()
    {
      _entity._local_index += 1;
      return *this;
    }

    bool operator==(const MeshIterator& other) const
    { return _entity._local_index == other._entity._local_index; }

    bool operator!=(const MeshIterator& other) const
    { return _entity._local_index != other._entity._local_index; }

    MeshEntity* operator->()
    { return &_entity; }

    MeshEntity& operator*()
    { return _entity; }

  private:

    // MeshEntity
    MeshEntity _entity;

  };

  /// An iterator for iterating over entities indicent to a MeshEntity
  // FIXME: Add 'MeshIterator MeshEntity::iterator(std::size_t dim);'?
  class MeshEntityIteratorNew
    : public std::iterator<std::forward_iterator_tag, MeshEntity>
  {
  public:

    // Constructor from MeshEntity and dimension
    MeshEntityIteratorNew(const MeshEntity& e, std::size_t dim,
                          std::size_t pos)
      : _entity(e.mesh(), dim, pos), _connections(nullptr)
    {
      // Get connectivity
      const MeshConnectivity& c = e.mesh().topology()(e.dim(), dim);

      // Compute connectivity if empty
      //if (c.empty())
      //  e.mesh().init(e.dim(), _entity.dim());

      // Pointer to array of connections
      _connections = c(e.index()) + pos;
    }

    /// Copy constructor
    MeshEntityIteratorNew(const MeshEntityIteratorNew& it)
      : _entity(it._entity), _connections(it._connections)
    {
      // Do nothing
    }

    // Copy assignment
    const MeshEntityIteratorNew& operator= (const MeshEntityIteratorNew& m)
    {
      _entity = m._entity;
      _connections = m._connections;
      return *this;
    }

    MeshEntityIteratorNew& operator++()
    {
      ++_connections;
      _entity._local_index = *_connections;
      return *this;
    }

    bool operator==(const MeshEntityIteratorNew& other) const
    { return _connections == other._connections; }

    bool operator!=(const MeshEntityIteratorNew& other) const
    { return _connections != other._connections; }

    MeshEntity* operator->()
    { return &_entity; }

    MeshEntity& operator*()
    { return _entity; }

  private:

    // MeshEntity
    MeshEntity _entity;

    // Array of connections for entity relative 'parent'
    const std::uint32_t* _connections;

  };

  /// Object with begin() and end() methods for iterating over
  /// entities incident to a MeshEntity
  // FIXME: Use consistent class name
  // Add method 'entities MeshEntity::items(std::size_t dim);'
  class entities
  {
  public:

    entities(const MeshEntity& e, int dim) : _entity(e), _dim(dim)
    {
      // Do nothing
    }

    const MeshEntityIteratorNew begin() const
    { return MeshEntityIteratorNew(_entity, _dim, 0); }

    MeshEntityIteratorNew begin()
    { return MeshEntityIteratorNew(_entity, _dim, 0); }

    const MeshEntityIteratorNew end() const
    {
      std::size_t n = _entity.num_entities(_dim);
      return MeshEntityIteratorNew(_entity, _dim, n);
    }

  private:

    const int _dim;
    const MeshEntity& _entity;
  };

  /// Object with begin() and end() methods for iterating over
  /// entities of a Mesh
  // FIXME: Use consistent class name
  // Add method 'entities Mesh::items(std::size_t dim);'
  class mesh_entities
  {
  public:

    mesh_entities(const Mesh& mesh, int dim) : _mesh(mesh), _dim(dim)
    {
      // Do nothing
    }

    const MeshIterator begin() const
    { return MeshIterator(_mesh, _dim, 0); }

    MeshIterator begin()
    { return MeshIterator(_mesh, _dim, 0); }

    const MeshIterator end() const
    { return MeshIterator(_mesh, _dim, _mesh.topology().ghost_offset(_dim)); }

  private:

    const Mesh& _mesh;
    const int _dim;

  };

}
