
#ifndef __MESH_ITERATOR_H
#define __MESH_ITERATOR_H

#include <memory>
#include <boost/iterator/iterator_facade.hpp>

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
  template<typename X> class entities;

  template<class T>
  class MeshIterator : public boost::iterator_facade<MeshIterator<T>, T, boost::forward_traversal_tag>
  {
  public:

    /// Default constructor
    MeshIterator() : _pos(0), _index(nullptr)
    {}

    /// Copy constructor
    MeshIterator(const MeshIterator& it) : _entity(std::make_unique<T>(it._entity->mesh(), 0)),
      _pos(it._pos), _index(it._index)
    {
      _entity->_local_index = (_index ? _index[_pos] : _pos);
    }

    // Copy assignment
    const MeshIterator& operator= (const MeshIterator& m)
    {
      _entity = std::make_unique<T>(*m._entity);
      _pos = m._pos;
      _index = m._index;
      return *this;
    }

    // Constructor with Mesh
    MeshIterator(const Mesh& mesh, std::size_t pos=0) : _pos(pos), _index(nullptr)
    {
      // Check if mesh is empty
      if (mesh.num_vertices() == 0)
        return;

      // Initialize mesh entity
      _entity = std::make_unique<T>(mesh, 0);
      _entity->_local_index = _pos;
    }

    // Constructor with MeshEntity
    MeshIterator(const MeshEntity& e, std::size_t pos=0) : _entity(std::make_unique<T>(e.mesh(), 0)), _pos(pos)
    {
      // Get connectivity
      const MeshConnectivity& c = e.mesh().topology()(e.dim(), _entity->dim());

      // Compute connectivity if empty
      if (c.empty())
        e.mesh().init(e.dim(), _entity->dim());

      // Set _index to point at connectivity for that entity
      _index = c(e.index());
      _entity->_local_index = _index[_pos];
    }

  private:

    friend class boost::iterator_core_access;

    void increment()
    {
      ++_pos;
      _entity->_local_index = (_index ? _index[_pos] : _pos);
    }

    bool equal(MeshIterator const& other) const
    {
      return (_pos == other._pos and _index == other._index);
    }

    T& dereference() const
    {
      return *_entity;
    }

    // MeshEntity
    std::unique_ptr<T> _entity;

    // Current position
    std::size_t _pos;

    // Mapping from pos to index (if any)
    const unsigned int* _index;

    template <typename X> friend class entities;
  };

  // Class defining begin() and end() methods for a given entity
  template<class T>
  class entities
  {
  public:
  entities(const Mesh& mesh) : _it_begin(mesh, 0)
    {
      const std::size_t dim = _it_begin._entity->dim();
      _it_end = MeshIterator<T>(mesh, mesh.topology().ghost_offset(dim));
    }

    entities(const MeshEntity& e) : _it_begin(e, 0)
    {
      const std::size_t dim = _it_begin._entity->dim();
      _it_end = MeshIterator<T>(e, e.num_entities(dim));
    }

    const MeshIterator<T> begin() const
    { return _it_begin; }

    const MeshIterator<T> end() const
    { return _it_end; }

  private:
    MeshIterator<T> _it_begin;
    MeshIterator<T> _it_end;
  };

  typedef entities<Cell> cells;
  typedef entities<Facet> facets;
  typedef entities<Face> faces;
  typedef entities<Edge> edges;
  typedef entities<Vertex> vertices;

}

#endif
