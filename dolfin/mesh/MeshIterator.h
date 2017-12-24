
#ifndef __MESH_ITERATOR_H
#define __MESH_ITERATOR_H

#include <memory>
#include <boost/iterator/iterator_facade.hpp>

#include "Mesh.h"
#include "MeshEntity.h"

namespace dolfin
{
  template<class T>
  class MeshIterator : public boost::iterator_facade<MeshIterator<T>, T,
    boost::forward_traversal_tag>
    {
    public:

    /// Default constructor
    MeshIterator() : _pos(0)
      {}

    // Copy constructor
    MeshIterator(const MeshIterator& it) : _pos(it._pos)
    {
      _entity->init(it._entity->mesh(), it._entity->dim(), _pos);
    }

    MeshIterator(const Mesh& mesh, std::size_t pos=0)
      : _pos(pos)
      {
        // Check if mesh is empty
        if (mesh.num_vertices() == 0)
          return;

        // Initialize mesh entity
        _entity->init(mesh, _entity->dim(), pos);
      }

    private:
      friend class boost::iterator_core_access;

      void increment()
      {
        ++_pos;
        _entity->_local_index = _pos;
      }

      bool equal(MeshIterator const& other) const
      {
        return (_pos == other._pos);
      }

      T& dereference() const
      {
        return *_entity;
      }

      // MeshEntity
      std::unique_ptr<T> _entity;

      // Current position
      std::size_t _pos;
    };

}

#endif
