#pragma once

#include <iterator>

#include "Cell.h"
#include "Mesh.h"
#include "MeshConnectivity.h"
#include "MeshEntity.h"
#include "MeshTopology.h"

namespace dolfin
{

  /// Iterator for entities of type T over a Mesh
  template<class T>
  class MeshIterator : public std::iterator<std::forward_iterator_tag, T>
  {
  public:

    /// Constructor for entities of dimension d
    MeshIterator(const Mesh& mesh, std::size_t dim, std::size_t pos) : _entity(mesh, dim, pos) {}

    /// Constructor for entities of dimension
    MeshIterator(const Mesh& mesh, std::size_t pos) : _entity(mesh, pos) {}

    /// Copy constructor
    MeshIterator(const MeshIterator& it) = default;

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
    T* operator->()
    { return &_entity; }

    /// Dereference
    T& operator*()
    { return _entity; }

    template<typename X> friend class MeshEntityRangeT;

 private:

    // MeshEntity
    T _entity;

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

      dolfin_assert(!c.empty());

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

  /// Iterator for entities of specified dimension that are incident to a MeshEntity
  template<class T>
  class MeshEntityIteratorNewT
    : public std::iterator<std::forward_iterator_tag, T>
  {
  public:

    // Constructor from MeshEntity and dimension
    MeshEntityIteratorNewT(const MeshEntity& e, std::size_t pos)
      : _entity(e.mesh(), pos), _connections(nullptr)
    {
      // Get connectivity
      const MeshConnectivity& c = e.mesh().topology()(e.dim(), _entity.dim());

      dolfin_assert(!c.empty());

      // Pointer to array of connections
      _connections = c(e.index()) + pos;
      _entity._local_index = *_connections;


      // Compute connectivity if empty
      //if (c.empty())
      //  e.mesh().init(e.dim(), _entity.dim());

      // FIXME: Handle case when number of attached entities is zero?
    }

    /// Copy constructor
    MeshEntityIteratorNewT(const MeshEntityIteratorNewT& it) = default;

    // Copy assignment
    const MeshEntityIteratorNewT& operator= (const MeshEntityIteratorNewT& m)
    {
      _entity = m._entity;
      _connections = m._connections;
      return *this;
    }

    MeshEntityIteratorNewT& operator++()
    { ++_connections; _entity._local_index = *_connections; return *this; }

    bool operator==(const MeshEntityIteratorNewT& other) const
    { return _connections == other._connections; }

    bool operator!=(const MeshEntityIteratorNewT& other) const
    { return _connections != other._connections; }

    T* operator->()
    { return &_entity; }

    T& operator*()
    { return _entity; }

    template<typename X> friend class EntityRangeT;

  private:

    // MeshEntity
    T _entity;

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

  // FIXME: handled ghosted meshes
  /// Class with begin() and end() methods for iterating over
  /// entities incident to a Mesh
  class MeshEntityRange
  {
  public:

    MeshEntityRange(const Mesh& mesh, int dim) : _mesh(mesh), _dim(dim) {}

    const MeshIterator<MeshEntity> begin() const
    { return MeshIterator<MeshEntity>(_mesh, _dim, 0); }

    MeshIterator<MeshEntity> begin()
    { return MeshIterator<MeshEntity>(_mesh, _dim, 0); }

    const MeshIterator<MeshEntity> end() const
    { return MeshIterator<MeshEntity>(_mesh, _dim, _mesh.topology().ghost_offset(_dim)); }

  private:

    const Mesh& _mesh;
    const int _dim;

  };

  template<class T>
  class MeshEntityRangeT
  {
  public:

    MeshEntityRangeT(const Mesh& mesh) : _mesh(mesh) {}

    const MeshIterator<T> begin() const
    { return MeshIterator<T>(_mesh, 0); }

    MeshIterator<T> begin()
    { return MeshIterator<T>(_mesh, 0); }

    const MeshIterator<T> end() const
    {
      auto it = MeshIterator<T>(_mesh, 0);
      std::size_t end = _mesh.topology().ghost_offset(it->dim());
      it->_local_index = end;
      return it;
      //return MeshIteratorT<T>(_mesh, _mesh.topology().ghost_offset(_dim));
    }

  private:

    const Mesh& _mesh;

  };

template<class T>
class EntityRangeT
  {
  public:

    EntityRangeT(const MeshEntity& e) : _entity(e) {}

    const MeshEntityIteratorNewT<T> begin() const
    { return MeshEntityIteratorNewT<T>(_entity, 0); }

    MeshEntityIteratorNewT<T> begin()
    { return MeshEntityIteratorNewT<T>(_entity, 0); }

    const MeshEntityIteratorNewT<T> end() const
    {
      auto it = MeshEntityIteratorNewT<T>(_entity, 0);
      std::size_t n = _entity.num_entities(it->dim());
      it._connections = it._connections + n;
      it->_local_index = *it._connections;
      return it;
      //return MeshEntityIteratorNewT<T>(_entity, n);
    }

  private:

    const MeshEntity& _entity;
    //const int _dim;

  };

  class Edge;
  class Facet;
  class Vertex;
  typedef EntityRangeT<Facet> FacetRange;
  typedef EntityRangeT<Vertex> VertexRange;
  typedef EntityRangeT<Edge> EdgeRange;

}
