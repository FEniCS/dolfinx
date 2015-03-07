// Copyright (C) 2011 Anders Logg
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
// First added:  2011-01-30
// Last changed: 2013-03-11

#ifndef __HIERARCHICAL_H
#define __HIERARCHICAL_H

#include <memory>

#include <dolfin/log/log.h>
#include <dolfin/log/LogStream.h>
#include "NoDeleter.h"

namespace dolfin
{

  /// This class provides storage and data access for hierarchical
  /// classes; that is, classes where an object may have a child
  /// and a parent.
  ///
  /// Note to developers: each subclass of Hierarchical that
  /// implements an assignment operator must call the base class
  /// assignment operator at the *end* of the subclass assignment
  /// operator. See the Mesh class for an example.

  template <typename T>
  class Hierarchical
  {
  public:

    /// Constructor
    Hierarchical(T& self) : _self(reference_to_no_delete_pointer(self)) {}

    /// Destructor
    virtual ~Hierarchical() {}

    /// Return depth of the hierarchy; that is, the total number of
    /// objects in the hierarchy linked to the current object via
    /// child-parent relationships, including the object itself.
    ///
    /// *Returns*
    ///     std::size_t
    ///         The depth of the hierarchy.
    std::size_t depth() const
    {
      std::size_t d = 1;
      for (std::shared_ptr<const T> it = root_node_shared_ptr();
           it->_child; it = it->_child)
        d++;
      return d;
    }

    /// Check if the object has a parent.
    ///
    /// *Returns*
    ///     bool
    ///         The return value is true iff the object has a parent.
    bool has_parent() const
    { return _parent ? true : false; }

    /// Check if the object has a child.
    ///
    /// *Returns*
    ///     bool
    ///         The return value is true iff the object has a child.
    bool has_child() const
    { return _child ? true : false; }

    /// Return parent in hierarchy. An error is thrown if the object
    /// has no parent.
    ///
    /// *Returns*
    ///     _Object_
    ///         The parent object.
    T& parent()
    {
      if (!_parent)
        dolfin_error("Hierarchical.h",
                     "extract parent of hierarchical object",
                     "Object has no parent in hierarchy");
      return *_parent;
    }

    /// Return parent in hierarchy (const version).
    const T& parent() const
    {
      if (!_parent)
        dolfin_error("Hierarchical.h",
                     "extract parent of hierarchical object",
                     "Object has no parent in hierarchy");
      return *_parent;
    }

    /// Return shared pointer to parent. A zero pointer is returned if
    /// the object has no parent.
    ///
    /// *Returns*
    ///     shared_ptr<T>
    ///         The parent object.
    std::shared_ptr<T> parent_shared_ptr()
    { return _parent; }

    /// Return shared pointer to parent (const version).
    std::shared_ptr<const T> parent_shared_ptr() const
    { return _parent; }

    /// Return child in hierarchy. An error is thrown if the object
    /// has no child.
    ///
    /// *Returns*
    ///     _T_
    ///         The child object.
    T& child()
    {
      if (!_child)
        dolfin_error("Hierarchical.h",
                     "extract child of hierarchical object",
                     "Object has no child in hierarchy");
      return *_child;
    }

    /// Return child in hierarchy (const version).
    const T& child() const
    {
      if (!_child)
        dolfin_error("Hierarchical.h",
                     "extract child of hierarchical object",
                     "Object has no child in hierarchy");
      return *_child;
    }

    /// Return shared pointer to child. A zero pointer is returned if
    /// the object has no child.
    ///
    /// *Returns*
    ///     shared_ptr<T>
    ///         The child object.
    std::shared_ptr<T> child_shared_ptr()
    { return _child; }

    /// Return shared pointer to child (const version).
    std::shared_ptr<const T> child_shared_ptr() const
    { return _child; }

    /// Return root node object in hierarchy.
    ///
    /// *Returns*
    ///     _T_
    ///         The root node object.
    T& root_node()
    {
      return *root_node_shared_ptr();
    }

    /// Return root node object in hierarchy (const version).
    const T& root_node() const
    {
      return *root_node_shared_ptr();
    }

    /// Return shared pointer to root node object in hierarchy.
    ///
    /// *Returns*
    ///     _T_
    ///         The root node object.
    std::shared_ptr<T> root_node_shared_ptr()
    {
      std::shared_ptr<T> it = _self;
      for (; it->_parent; it = it->_parent);
      return it;
    }

    /// Return shared pointer to root node object in hierarchy (const version).
    std::shared_ptr<const T> root_node_shared_ptr() const
    {
      std::shared_ptr<const T> it = _self;
      for (; it->_parent; it = it->_parent);
      return it;
    }

    /// Return leaf node object in hierarchy.
    ///
    /// *Returns*
    ///     _T_
    ///         The leaf node object.
    T& leaf_node()
    {
      return *leaf_node_shared_ptr();
    }

    /// Return leaf node object in hierarchy (const version).
    const T& leaf_node() const
    {
      return *leaf_node_shared_ptr();
    }

    /// Return shared pointer to leaf node object in hierarchy.
    ///
    /// *Returns*
    ///     _T_
    ///         The leaf node object.
    std::shared_ptr<T> leaf_node_shared_ptr()
    {
      std::shared_ptr<T> it = _self;
      for (; it->_child; it = it->_child);
      return it;
    }

    /// Return shared pointer to leaf node object in hierarchy (const version).
    std::shared_ptr<const T> leaf_node_shared_ptr() const
    {
      std::shared_ptr<const T> it = _self;
      for (; it->_child; it = it->_child);
      return it;
    }

    /// Set parent
    void set_parent(std::shared_ptr<T> parent)
    { _parent = parent; }

    /// Clear child
    void clear_child()
    {
      _child.reset();
    }

    /// Set child
    void set_child(std::shared_ptr<T> child)
    { _child = child; }

    /// Assignment operator
    const Hierarchical& operator= (const Hierarchical& hierarchical)
    {
      // Destroy any previous parent-child relations
      _parent.reset();
      _child.reset();

      return *this;
    }

    /// Function useful for debugging the hierarchy
    void _debug() const
    {
      info("Debugging hierarchical object:");
      cout << "  depth           = " << depth() << endl;
      cout << "  has_parent()    = " << has_parent() << endl;
      info("  _parent.get()   = %x", _parent.get());
      info("  _parent.count() = %d", _parent.use_count());
      cout << "  has_child()     = " << has_parent() << endl;
      info("  _child.get()    = %x", _parent.get());
      info("  _child.count()  = %d", _parent.use_count());
    }

  private:

    // The object itself
    std::shared_ptr<T> _self;

    // Parent and child in hierarchy
    std::shared_ptr<T> _parent;
    std::shared_ptr<T> _child;

  };

}

#endif
