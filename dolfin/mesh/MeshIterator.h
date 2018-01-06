
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
      //std::cout << "    Ent empty constructor" << std::endl;
      // Do nothing
    }

    /// Copy constructor
    MeshIterator(const MeshIterator& it) : _entity(it._entity),  _pos(it._pos),
      _connections(it._connections)
    {
      //std::cout << "    Ent copy: " << it._pos << std::endl;
      // Do nothing
    }

    // Copy assignment
    const MeshIterator& operator= (const MeshIterator& m)
    {
      //std::cout << "    Ent assign" << std::endl;
      _entity = m._entity;
      _pos = m._pos;
      _connections = m._connections;
      return *this;
    }

    // Constructor with Mesh
    MeshIterator(const Mesh& mesh, std::size_t pos=0)
      : _entity(mesh, 0), _pos(pos), _connections(nullptr)
    {
      //std::cout << "    Ent assign" << std::endl;
      // Check if mesh is empty
      //if (mesh.num_vertices() == 0)
      //  return;
    }

    // Constructor with MeshEntity
    MeshIterator(const MeshEntity& e, std::size_t pos=0)
      : _entity(e.mesh(), 0), _pos(pos)
    {
      //std::cout << "    Ent constructor" << std::endl;

      // Get connectivity
      const MeshConnectivity& c = e.mesh().topology()(e.dim(), _entity.dim());

      // Compute connectivity if empty
      //if (c.empty())
      //  e.mesh().init(e.dim(), _entity.dim());

      // Pointer to array of connections
      _connections = c(e.index());
    }

    MeshIterator& operator++()
    {
      //std::cout << "    Ent incr" << std::endl;
      ++_pos;
      return *this;
    }

    bool operator==(const MeshIterator& other) const
    {
      //std::cout << "    Ent ==" << std::endl;
      return _pos == other._pos;
    }

    bool operator!=(const MeshIterator& other) const
    {
      //std::cout << "    Ent !=" << std::endl;
      return _pos != other._pos;
    }

    T* operator->()
    {
      _entity._local_index = (_connections ? _connections[_pos] : _pos);
      return &_entity;
    }

    T& operator*()
    {
      _entity._local_index = (_connections ? _connections[_pos] : _pos);
      return _entity;
    }

  private:

    // MeshEntity
    T _entity;

    // Current position
    std::size_t _pos;

    // Array of connections for entity relative 'parent'
    const unsigned int* _connections;

    template <typename X> friend class entities;
  };

  // Class defining begin() and end() methods for a given entity
  template<class T>
    class entities
  {
  public:

  entities(const MeshEntity& e, int dim) : _entity(e), _dim(dim)
    {
      // Do nothing
    }

    const MeshIterator<T> begin() const
    {
      return MeshIterator<T>(_entity, 0);
    }

    MeshIterator<T> begin()
    {
      return MeshIterator<T>(_entity, 0);
    }

    const MeshIterator<T> end() const
    {
      std::size_t n = _entity.num_entities(_dim);
      return MeshIterator<T>(_entity, n);
    }

  private:

    const int _dim;
    const MeshEntity& _entity;
  };

  typedef entities<Cell> cells;
  typedef entities<Facet> facets;
  typedef entities<Face> faces;
  typedef entities<Edge> edges;
  typedef entities<Vertex> vertices;


  // Class defining begin() and end() methods for a given entity
  template<typename T> class mesh_entities
  {
  public:

    mesh_entities(const Mesh& mesh, int dim) : _dim(dim), _mesh(mesh)
    {
      // Do nothing
    }

    const MeshIterator<T> begin() const
    { return MeshIterator<T>(_mesh); }

    MeshIterator<T> begin()
    { return MeshIterator<T>(_mesh); }

    const MeshIterator<T> end() const
    { return MeshIterator<T>(_mesh, _mesh.topology().ghost_offset(_dim)); }

  private:

    const int _dim;
    const Mesh& _mesh;
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
