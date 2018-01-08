#pragma once

#include <iterator>

#include "Mesh.h"
#include "MeshConnectivity.h"
#include "MeshEntity.h"
#include "MeshTopology.h"

namespace dolfin
{

  // FIXME: Add 'MeshIterator Mesh::iterator(std::size_t dim);'?

  /// Iterator for entities of a given dimension of a Mesh
  class MeshIterator : public std::iterator<std::forward_iterator_tag, MeshEntity>
  {
  public:

    /// Constructor for entitirs off dimension d
    MeshIterator(const Mesh& mesh, std::size_t dim, std::size_t pos)
      : _entity(mesh, dim, pos) { /* Do nothing */ }

    /// Copy constructor
    MeshIterator(const MeshIterator& it) = default;
    /*
    MeshIterator(const MeshIterator& it) : _entity(it._entity)
    {
      // Do nothing
    }
    */

    /// Copy assignment
    const MeshIterator& operator= (const MeshIterator& m)
    { _entity = m._entity; return *this; }

    /// Increment iterator
    MeshIterator& operator++()
    { _entity._local_index += 1; return *this; }

    /// Return true if equal
    bool operator==(const MeshIterator& other) const
    { return _entity._local_index == other._entity._local_index; }

    /// Return true if not equal
    bool operator!=(const MeshIterator& other) const
    { return _entity._local_index != other._entity._local_index; }

    /// Member access
    MeshEntity* operator->()
    { return &_entity; }

    /// Deference
    MeshEntity& operator*()
    { return _entity; }

  private:

    // MeshEntity
    MeshEntity _entity;

  };

  // FIXME: Add 'MeshIterator MeshEntity::iterator(std::size_t dim);'?

  /// Iterator for entities of specified dimension that are incident to a MeshEntity
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

      dolfin_assert(!c.empty())

      // Pointer to array of connections
      _connections = c(e.index()) + pos;
      _entity._local_index = *_connections;


      // Compute connectivity if empty
      //if (c.empty())
      //  e.mesh().init(e.dim(), _entity.dim());

      // FIXME: Handle case when number of attached entities is zero?
    }

    /// Copy constructor
    MeshEntityIteratorNew(const MeshEntityIteratorNew& it) = default;

    // Copy assignment
    const MeshEntityIteratorNew& operator= (const MeshEntityIteratorNew& m)
    {
      _entity = m._entity;
      _connections = m._connections;
      return *this;
    }

    MeshEntityIteratorNew& operator++()
    { ++_connections; _entity._local_index = *_connections; return *this; }

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

    // Pointer to current entity index
    const std::uint32_t* _connections;

  };

  // FIXME: Add method 'entities MeshEntity::items(std::size_t dim);'

  /// Class with begin() and end() methods for iterating over
  /// entities incident to a MeshEntity
  class EntityRange
  {
  public:

    EntityRange(const MeshEntity& e, int dim) : _entity(e), _dim(dim) {}

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

    const MeshEntity& _entity;
    const int _dim;

  };

  /// Class with begin() and end() methods for iterating over
  /// entities incident to a MeshEntity
  class MeshEntityRange
  {
  public:

    MeshEntityRange(const Mesh& mesh, int dim) : _mesh(mesh), _dim(dim)
    { /* Do nothing */ }

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
