// Copyright (C) 2011 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-01-30
// Last changed: 2011-01-31

#ifndef __HIERARCHICAL_H
#define __HIERARCHICAL_H

#include <boost/shared_ptr.hpp>
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

    /// Check if the object has a parent.
    ///
    /// *Returns*
    ///     bool
    ///         The return value is true iff the object has a parent.
    bool has_parent() const
    { return _parent != 0; }

    /// Check if the object has a child.
    ///
    /// *Returns*
    ///     bool
    ///         The return value is true iff the object has a child.
    bool has_child() const
    { return _child != 0; }

    /// Return parent in hierarchy. An error is thrown if the object
    /// has no parent.
    ///
    /// *Returns*
    ///     _Object_
    ///         The parent object.
    T& parent() const
    {
      if (!has_parent())
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
      if (!has_child())
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
    {
      return *coarse_shared_ptr();
    }

    /// Return shared pointer to coarsest object in hierarchy.
    ///
    /// *Returns*
    ///     _T_
    ///         The coarse object.
    boost::shared_ptr<T> coarse_shared_ptr()
    {
      // Some trixing to handle the case when <this> is the itself the
      // coarse object (can't be converted to type T).
      if (!has_parent() && !has_child())
      {
        error("Hierarchy is empty (only one object in hierarchy).");
        return boost::shared_ptr<T>();
      }
      else if (!has_parent())
        return child().coarse_shared_ptr();

      // Find parent of parent of parent of...
      boost::shared_ptr<T> object = parent_shared_ptr();
      for (; object->has_parent(); object = object->parent_shared_ptr());
      return object;
    }

    /// Return finest object in hierarchy.
    ///
    /// *Returns*
    ///     _T_
    ///         The fine object.
    T& fine()
    {
      return *fine_shared_ptr();
    }

    /// Return shared pointer to finest object in hierarchy.
    ///
    /// *Returns*
    ///     _T_
    ///         The fine object.
    boost::shared_ptr<T> fine_shared_ptr()
    {
      // Some trixing to handle the case when <this> is the itself the
      // fine object (can't be converted to type T).
      if (!has_parent() && !has_child())
      {
        error("Hierarchy is empty (only one object in hierarchy).");
        return boost::shared_ptr<T>();
      }
      else if (!has_child())
        return parent().fine_shared_ptr();

      // Find child of child of child of...
      boost::shared_ptr<T> object = child_shared_ptr();
      for (; object->has_child(); object = object->child_shared_ptr());
      return object;
    }

    /// Set parent
    void set_parent(boost::shared_ptr<T> parent)
    { _parent = parent; }

    /// Set child
    void set_child(boost::shared_ptr<T> child)
    { _child = child; }

  private:

    // Parent and child in hierarchy
    mutable boost::shared_ptr<T> _parent;
    mutable boost::shared_ptr<T> _child;

  };

}

#endif
