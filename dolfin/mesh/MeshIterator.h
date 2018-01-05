
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
  template<typename X> class entities;

  template<class T>
    class MeshIterator : public std::iterator<std::forward_iterator_tag, T>
  {
  public:

    // Default constructor
    MeshIterator()
    {
      // Do nothing
    }

    /// Copy constructor
    MeshIterator(const MeshIterator& it) : _entity(it._entity),  _pos(it._pos)
    {
      // Do nothing
    }

    // Copy assignment
    const MeshIterator& operator= (const MeshIterator& m)
    {
      _entity = m._entity;
      _pos = m._pos;
      return *this;
    }

    // Constructor with Mesh
    MeshIterator(const Mesh& mesh, std::size_t pos=0)
      : _entity(mesh, 0), _pos(pos)
    {
      // Check if mesh is empty
      //if (mesh.num_vertices() == 0)
      //  return;
    }

    // Constructor with MeshEntity
    MeshIterator(const MeshEntity& e, std::size_t pos=0)
      : _entity(e.mesh(), 0), _pos(pos)
    {
      // Get connectivity
      const MeshConnectivity& c = e.mesh().topology()(e.dim(), _entity.dim());

      // Compute connectivity if empty
      if (c.empty())
        e.mesh().init(e.dim(), _entity.dim());
    }

    MeshIterator& operator++()
    {
      ++_pos;
      return *this;
    }

    bool operator==(const MeshIterator& other) const
    {
      return (_pos == other._pos);
    }

    bool operator!=(const MeshIterator& other) const
    {
      return (_pos != other._pos);
    }

    T* operator->()
    {
      _entity._local_index = _pos;
      return &_entity;
    }

    T& operator*()
    {
      _entity._local_index = _pos;
      return _entity;
    }

  private:

    template <typename X> friend class entities;

    // MeshEntity
    T _entity;

    // Current position
    std::size_t _pos;
  };

  /// Class defining begin() and end() methods for a given entity
  template<class T> class entities
  {
  public:

    entities(const Mesh& mesh) : _it_begin(mesh, 0) //, _it_end(_it_begin)
    {
      // Don't initialise mesh or entity for end iterator
      const std::size_t dim = _it_begin._entity.dim();
      _it_end._pos = mesh.topology().ghost_offset(dim);
    }

    entities(const MeshEntity& e) : _it_begin(e, 0) //, _it_end(_it_begin)
    {
      // Don't bother initialising mesh or entity for end iterator
      const std::size_t dim = _it_begin._entity.dim();
      _it_end._pos = e.num_entities(dim);
    }

    const MeshIterator<T>& begin() const
    { return _it_begin; }

    MeshIterator<T>& begin()
    { return _it_begin; }

    const MeshIterator<T>& end() const
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
