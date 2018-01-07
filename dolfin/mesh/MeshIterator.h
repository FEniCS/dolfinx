
#ifndef __MESH_ITERATOR_H
#define __MESH_ITERATOR_H

#include <memory>
#include <iterator>

#include "Mesh.h"
#include "MeshEntity.h"

namespace dolfin
{
  // Forward declarations
  class Vertex;
  class Edge;
  class Face;
  class Facet;
  class Cell;
  //template<typename X> class entities;

  /// An iterator for iterating over entities of a Mesh
  // FIXME: Add 'MeshIterator Mesh::iterator(std::size_t dim);'?
  // FIXME: Shouldn't really template this class over cell type
  //template<class T>
    class MeshIterator : public std::iterator<std::forward_iterator_tag,
      MeshEntity>
  {
  public:

    // Constructor from Mesh and entity dimension
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

    // Copy assignment
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
  //template<class T>
    class MeshEntityIteratorNew
    : public std::iterator<std::forward_iterator_tag, MeshEntity>
  {
  public:

    // Constructor from MeshEntity and dimension
    MeshEntityIteratorNew(const MeshEntity& e, std::size_t dim,
                          std::size_t pos)
      : _entity(e.mesh(), dim, pos), _pos(pos), _connections(nullptr)
    {
      // Get connectivity
      const MeshConnectivity& c = e.mesh().topology()(e.dim(), dim);

      // Compute connectivity if empty
      //if (c.empty())
      //  e.mesh().init(e.dim(), _entity.dim());

      // Pointer to array of connections
      _connections = c(e.index());
    }

    /// Copy constructor
    MeshEntityIteratorNew(const MeshEntityIteratorNew& it)
      : _entity(it._entity), _pos(it._pos), _connections(it._connections)
    {
      // Do nothing
    }

    // Copy assignment
    const MeshEntityIteratorNew& operator= (const MeshEntityIteratorNew& m)
    {
      _entity = m._entity;
      _pos = m._pos;
      _connections = m._connections;
      return *this;
    }

    MeshEntityIteratorNew& operator++()
    {
      ++_pos;
      return *this;
    }

    bool operator==(const MeshEntityIteratorNew& other) const
    { return _pos == other._pos; }

    bool operator!=(const MeshEntityIteratorNew& other) const
    { return _pos != other._pos; }

    MeshEntity* operator->()
    {
      _entity._local_index = _connections[_pos];
      return &_entity;
    }

    MeshEntity& operator*()
    {
      _entity._local_index = _connections[_pos];
      return _entity;
    }

  private:

    // MeshEntity
    MeshEntity _entity;

    // Current position
    std::size_t _pos;

    // Array of connections for entity relative 'parent'
    const unsigned int* _connections;

  };

  /// Object with begin() and end() methods for iterating over
  /// entities incident to a MeshEntity
  // FIXME: Use consistent class name
  // Add method 'entities MeshEntity::items(std::size_t dim);'
  //template<class T>
    class entities
  {
  public:

    entities(const MeshEntity& e, int dim) : _entity(e), _dim(dim)
    {
      // Do nothing
    }

    const MeshEntityIteratorNew begin() const
    {
      return MeshEntityIteratorNew(_entity, _dim, 0);
    }

    MeshEntityIteratorNew begin()
    {
      return MeshEntityIteratorNew(_entity, _dim, 0);
    }

    const MeshEntityIteratorNew end() const
    {
      std::size_t n = _entity.num_entities(_dim);
      return MeshEntityIteratorNew(_entity, _dim, n);
    }

  private:

    const int _dim;
    const MeshEntity& _entity;
  };

  /*
  typedef entities<Cell> cells;
  typedef entities<Facet> facets;
  typedef entities<Face> faces;
  typedef entities<Edge> edges;
  typedef entities<Vertex> vertices;
  */

  /// Object with begin() and end() methods for iterating over
  /// entities of a Mesh
  // FIXME: Use consistent class name
  // Add method 'entities Mesh::items(std::size_t dim);'
  //template<typename T>
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

  /*
  typedef mesh_entities<Cell> mesh_cells;
  //typedef mesh_entities<Facet> mesh_facets;
  typedef mesh_entities<Face, 2> mesh_faces;
  typedef mesh_entities<Edge, 1> mesh_edges;
  typedef mesh_entities<Vertex, 0> mesh_vertices;
  */

}

#endif
