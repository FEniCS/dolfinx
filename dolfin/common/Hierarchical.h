// Copyright (C) 2011 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-01-30
// Last changed: 2011-02-02

#ifndef __HIERARCHICAL_H
#define __HIERARCHICAL_H

#include <boost/shared_ptr.hpp>

#include <dolfin/log/dolfin_log.h>
#include "NoDeleter.h"

namespace dolfin
{

  /// This class provides storage and data access for hierarchical
  /// classes; that is, classes where an object may have a child
  /// and a parent.

  template<class T>
  class Hierarchical
  {
  public:

    Hierarchical()
    {
      _debug();
    }

    /// Return depth of the hierarchy; that is, the total number of
    /// objects in the hierarchy linked to the current object via
    /// child-parent relationships, including the object itself.
    ///
    /// *Returns*
    ///     uint
    ///         The depth of the hierarchy.
    uint depth() const
    {
      // Depth is 1 if there is no parent nor child
      if (!_parent && !_child)
        return 1;

      // If we have a parent, step back to coarse and then to fine
      if (_parent)
      {
        boost::shared_ptr<T> it = _parent;
        for (; it->_parent; it = it->_parent);
        uint d = 1;
        for (; it->_child; it = it->_child) d++;
        return d;
      }

      // If we have a child, step to fine and then to coarse
      if (_child)
      {
        boost::shared_ptr<T> it = _child;
        for (; it->_child; it = it->_child);
        uint d = 1;
        for (; it->_parent; it = it->_parent) d++;
        return d;
      }

      // Please compiler
      return 0;
    }

    /// Check if the object has a parent.
    ///
    /// *Returns*
    ///     bool
    ///         The return value is true iff the object has a parent.
    bool has_parent() const
    { return _parent; }

    /// Check if the object has a child.
    ///
    /// *Returns*
    ///     bool
    ///         The return value is true iff the object has a child.
    bool has_child() const
    { return _child; }

    /// Return parent in hierarchy. An error is thrown if the object
    /// has no parent.
    ///
    /// *Returns*
    ///     _Object_
    ///         The parent object.
    T& parent() const
    {
      if (!_parent)
        error("Object has no parent in hierarchy.");
      return *_parent;
    }

    /// Return shared pointer to parent. A zero pointer is returned if
    /// the object has no parent.
    ///
    /// *Returns*
    ///     shared_ptr<T>
    ///         The parent object.
    boost::shared_ptr<T> parent_shared_ptr() const
    { return _parent; }

    /// Return child in hierarchy. An error is thrown if the object
    /// has no child.
    ///
    /// *Returns*
    ///     _T_
    ///         The child object.
    T& child() const
    {
      if (!_child)
        error("Object has no child in hierarchy.");
      return *_child;
    }

    /// Return shared pointer to child. A zero pointer is returned if
    /// the object has no child.
    ///
    /// *Returns*
    ///     shared_ptr<T>
    ///         The child object.
    boost::shared_ptr<T> child_shared_ptr() const
    { return _child; }

    /// Return coarsest object in hierarchy.
    ///
    /// *Returns*
    ///     _T_
    ///         The coarse object.
    T& coarse()
    { return *coarse_shared_ptr(); }

    /// Return shared pointer to coarsest object in hierarchy.
    ///
    /// *Returns*
    ///     _T_
    ///         The coarse object.
    boost::shared_ptr<T> coarse_shared_ptr()
    {
      // Some trixing to handle the case when <this> is the itself the
      // coarse object (can't be converted to type T).
      if (!_parent && !_child)
      {
        error("Unable to return coarse object, hierarchy is empty (only one object in hierarchy).");
        return boost::shared_ptr<T>();
      }
      else if (!_parent)
        return _child->coarse_shared_ptr();

      // Find parent of parent of parent of...
      boost::shared_ptr<T> it = _parent;
      for (; it->_parent; it = it->_parent);
      return it;
    }

    /// Return finest object in hierarchy.
    ///
    /// *Returns*
    ///     _T_
    ///         The fine object.
    T& fine()
    { return *fine_shared_ptr(); }

    /// Return shared pointer to finest object in hierarchy.
    ///
    /// *Returns*
    ///     _T_
    ///         The fine object.
    boost::shared_ptr<T> fine_shared_ptr()
    {
      // Some trixing to handle the case when <this> is the itself the
      // fine object (can't be converted to type T).
      if (!_parent && !_child)
      {
        error("Unable to return fine object, hierarchy is empty (only one object in hierarchy).");
        return boost::shared_ptr<T>();
      }
      else if (!_child)
        return _parent->fine_shared_ptr();

      // Find child of child of child of...
      boost::shared_ptr<T> it = _child;
      for (; it->_child; it = it->_child);
      return it;
    }

    /// Set parent
    void set_parent(boost::shared_ptr<T> parent)
    { _parent = parent; }

    /// Set child
    void set_child(boost::shared_ptr<T> child)
    { _child = child; }

    /// Assignment operator
    const Hierarchical& operator= (const Hierarchical& hierarchical)
    {
      // Assignment to object part of a hierarchy not allowed as it
      // would either require a very exhaustive logic to handle or
      // completely mess up child-parent relations and data ownership.
      const uint d = depth();
      if (d > 1)
      {
        error("Unable to assign, object is part of a hierarchy (depth = %d).", d);
      }

      return *this;
    }

    /// Function useful for debugging the hierarchy
    void _debug() const
    {
      info("Debugging hierarchical object.");
      cout << "  depth           = " << depth() << endl;
      cout << "  has_parent()    = " << has_parent() << endl;
      info("  _parent.get()   = %x", _parent.get());
      info("  _parent.count() = %d", _parent.use_count());
      cout << "  has_child()     = " << has_parent() << endl;
      info("  _child.get()    = %x", _parent.get());
      info("  _child.count()  = %d", _parent.use_count());
    }

  private:

    // Parent and child in hierarchy
    mutable boost::shared_ptr<T> _parent;
    mutable boost::shared_ptr<T> _child;

  };

}

#endif
