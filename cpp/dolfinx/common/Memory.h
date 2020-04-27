// Copyright (C) 2020 Matthias Rambausek
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cassert>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <vector>
#include <list>

namespace dolfinx::common::memory
{

using lock_t = std::shared_ptr<const bool>;
using sentinel_t = std::weak_ptr<const bool>;

// TODO: either use a signal/slot lib or rely on weak/shared_ptr without
//  callbacks.

/// A wrapper for objects with lifetime dependencies without using smart
/// pointers but mimicking their interface to a certain degree.
template <typename T, typename observer_t>
class observed
{
public:
  /// Constructor: taking the object and lifetime information (lock or
  /// "sentinel")
  observed(T&& obj, observer_t observer)
      : obj{std::move(obj)}, observer{std::move(observer)}
  {
    // do nothing
  }

  /// Default constructor
  observed() = default;

  /// Default copy
  observed(const observed& other) = default;

  /// Default move
  explicit observed(observed&& other) noexcept = default;

  /// Default copy assignemnt
  observed& operator=(const observed& other) = default;

  /// Default move assignemnt
  observed& operator=(observed&& other) noexcept = default;

  /// Default constructor
  ~observed() = default;

  /// Get reference on the object stored
  T& get()
  {
    assert(use_count() != 0);
    return obj;
  }

  /// Get const reference on the object stored
  const T& get() const
  {
    assert(use_count() != 0);
    return obj;
  }

  /// Use count of the observer
  [[nodiscard]] long use_count() const noexcept { return observer.use_count(); }

  /// Reset the observer
  void reset() noexcept { observer.reset(); }

private:
  T obj;
  observer_t observer;
};

template <typename T>
using maybe_null = T*;

template <typename T>
using weakly_owned = observed<T, sentinel_t>;

template <typename T>
using weakly_owned = observed<T, sentinel_t>;

/// Make an object weakly owned (with lifetime information 'sentinel')
template <typename T>
weakly_owned<T> make_guarded(T obj, sentinel_t sentinel)
{
  return weakly_owned<T>{std::forward<T&&>(obj), std::move(sentinel)};
}

/// Create whether the object is still alive/valid
template <typename T>
bool check(const weakly_owned<T>& obj)
{
  return obj.use_count() != 0;
}

/// Create whether the object is still alive/valid: overload for pointers.
template <typename T>
bool check(const weakly_owned<T*>& obj)
{
  return obj.use_count() != 0 && obj.get();
}

/// An owned object: the (shared) owner hols also the attached lifetime
/// information such that a non-owner can detect its status, e.g. weakly own and
/// safely use or not use it.
template <typename T>
using owned = observed<T, lock_t>;

/// Attach a "liftime" to an object.
template <typename T>
owned<T> make_owned(T obj, lock_t lock)
{
  return owned<T>{std::forward<T&&>(obj), std::move(lock)};
}

// template <typename Iterator_t, typename Visitor_t>
// auto visit(Iterator_t begin, Iterator_t end, Visitor_t&& visitor)
//{
//  decltype(visitor(begin->get())) res;
//  for (auto layer = begin; layer != end; ++layer)
//  {
//    assert(check(*layer));
//    if (res = visitor(layer->get()); res)
//      return res;
//  }
//  return res;
//}

/// Apply a visitor to each element of a range until the visitor returns true
/// or the end of the range is reached. Additional args for the visitor can be
/// provided.
template <typename Iterator_t, typename Visitor_t, typename... Args_t>
auto visit(Iterator_t begin, Iterator_t end, Visitor_t&& visitor,
           Args_t&&... args)
{
  decltype(visitor(begin->get(), std::forward<Args_t>(args)...)) res;
  for (auto layer = begin; layer != end; ++layer)
  {
    assert(check(*layer));
    if (res = visitor(layer->get(), std::forward<Args_t>(args)...); res)
      return res;
  }
  return res;
}

template <typename Layer_t>
class LayerManager;

/// A class holding a layer from a layer manager and by that determining its
/// lifetime. If the lock is destroyed it will give notice to the layer_manager.
/// A layer can be owned by more than one lock (shared ownership). The lock
/// can expose the layer for inspection.
template <typename Layer_t>
class LayerLock
{
public:
  /// Constructor: require a reference to the guarded layer and a weakly owned
  /// callback that is invoked when the lock is released or destroyed.
  LayerLock(owned<maybe_null<const Layer_t>> layer,
            weakly_owned<std::function<void(maybe_null<const Layer_t>)>>
                on_destruction)
      : _layer{std::move(layer)}, _on_destruction{std::move(on_destruction)}
  {
    // do nothing
  }

  /// Default copy constructor
  LayerLock(const LayerLock&) = default;

  /// Default move constructor
  LayerLock(LayerLock&&) noexcept = default;

  /// Default copy assignments
  LayerLock& operator=(const LayerLock&) = default;

  /// Default move assignment
  LayerLock& operator=(LayerLock&&) noexcept = default;

  /// Destructor: do release operation (notify that the lock has gone)
  ~LayerLock() { release(); };

  /// Returns a pointer to the owned layer. If no layer os owned because
  /// the lock has been released.
  maybe_null<const Layer_t> layer() const
  {
    // Check if the user of the object is still there and return nullptr if not
    return check(_on_destruction) ? maybe_null<const Layer_t>{_layer.get()}
                                  : maybe_null<const Layer_t>{nullptr};
  }

  /// Relases ownership.
  void release()
  {
    // Retrieve pointer on layer
    maybe_null<const Layer_t> layer_ptr = layer();
    // Give it free for cleanup
    _layer.reset();
    // Signalize to the layer manager that a lock has gone and tell the address
    // of the layer
    if (check(_on_destruction))
      _on_destruction.get()(layer_ptr);
  };

private:
  // Owned layer
  owned<const Layer_t*> _layer;

  // Callback informing that the lock has been released
  weakly_owned<std::function<void(maybe_null<const Layer_t>)>> _on_destruction;
};

// NOTE concerning multithreading: there is a lot to optimize there. The layer
// structure allows, eg., for reading everything below the active layer at
// any time such that new write layer can be created on top without blocking
// the read locks or being blocked by them. The current implementation
// is quite coarse with the SafeHandle only giving write access if no one
// else holds a lock (shared/unique ownership of a shared lock!).
// However, multithreading is not a typical scenario in the present scope
// and, moreover, read/write operations are not performance critical here.
// Thus, to keep the code clean, not much work is done in that direction
// and what is in place might be removed.

// NOTE concerning copies: It is difficult to ask how to copy this data
// structure. It is not clear what kind of copy should be made. Therefore,
// since all essential data can be accessed, it is up to the user to create
// a new instance transferred the data he wants to preserve.

// NOTE on memory management: The current model is a stack, that means that
// data is not necessarily cleaned up when its handle dies but only when no
// other data depends on it any longer. This behavior, however, can be adapted
// easily. Related to that, cleanup is triggered each time a layer lock goes
// ends its life. There is no kind of automatic cleanup. However, that could be
// easily added as well. Two general types of behavior:
// 1. Stack (current) == last in / first out, cleanup possibly delayed until
//    dependent object go away.
// 2. Immediate cleanup, pull out layers from the middle of the layer stack
// If the layer locks are not passed around as crazy, both approaches are
// equivalent. In general, it is quite straighforward to switch, unless one
// aims for sophisticated multithreading. Also, both variants can be base on
// callbacks and/or regularly (each access) triggered cleanup.
// Regarding the implementation: The current implementation does not rely on
// smart pointers simply for constness reasons. The wrapper types employed
// instead behave quite similar, however, such they can easily be interchanged.

/// Manages a collection of data layers. It provides means to add layers,
/// read and write them.
template <typename Layer_t>
class LayerManager
{
public:

  /// Alias for the layer lock type
  using LayerLock_t = LayerLock<Layer_t>;

  /// Create Storage layer
  /// @param[in] remanent if given creates a remanent storage layer of which the
  /// lifetime is bound to this object.
  /// @param[in] other points to another LayerManager for read only access.
  /// Its data will not be copied into the new LayerManager.
  LayerManager(bool remanent, const LayerManager& other)
      : other{make_guarded(std::cref(other), other.lifetime)}
  {
    if (remanent)
      remanent_lock.emplace(acquire_layer_lock(true));
  }

  /// Constructor without background storage
  LayerManager(bool remanent)
  {
    if (remanent)
      remanent_lock.emplace(acquire_layer_lock(true));
  }

  /// Default constructor: No remanent layer, no background storage
  LayerManager() = default;

  /// Copy constructor [deleted]
  LayerManager(const LayerManager& other) = delete;

  /// Move constructor
  explicit LayerManager(LayerManager&& other) noexcept = default;

  /// Destructor
  ~LayerManager() = default;

  /// Copy assignment [deleted]
  LayerManager& operator=(const LayerManager& other) = delete;

  /// Move assignment
  LayerManager& operator=(LayerManager&& other) noexcept = default;

  /// Create a new lock-like handle to keep a storage layer alive. By default,
  /// this returns just is another handle for the current layer. The creation of
  /// a new write layer is optional. Note that layers will be destroyed in
  /// reverse order of creation. This means, they do not immediately vanish if
  /// the locks leave their scope [easy to change, depends on whether this is a
  /// valid use case at all].
  /// Once a new layer is active, lower layers cannot be written to. This
  /// creates the possibility to recover the previous state. Also not that
  /// while lower layers cannot be written to, they can be read.
  /// However, that the read-only "other" data may still be changed from
  /// outside.
  LayerLock_t acquire_layer_lock(bool force_new_layer = false)
  {
    auto layer_lock = std::make_shared<const bool>(true);
    if (layers.empty() or force_new_layer)
    {
      layers.emplace_back(Layer_t{}, layer_lock);
    }
    std::function<void(maybe_null<const Layer_t> layer)> callback
        = [=](maybe_null<const Layer_t> layer) { this->layer_expired(layer); };
    return LayerLock_t(make_owned<maybe_null<const Layer_t>>(
                           &active_layer().get(), std::move(layer_lock)),
                       make_guarded(callback, lifetime));
  }

  /// Add a whole new layer from outside
  void push_layer(Layer_t layer) { layers.emplace_back(std::move(layer)); }

  /// Ask whether the storage has a remanent layer
  bool is_remanent() { return remanent_lock.has_value(); }

  /// Ask whether the storage is empty, i.e. does not have ana writable layers.
  bool empty() const { return layers.empty(); }

  /// Ask whether the storage has read-only background storage attached.
  bool has_other() const { return check_other(); }

  /// Get the attached read-only storage
  const std::optional<weakly_owned<std::reference_wrapper<const LayerManager>>>& get_other() const { return other; }

  /// Access alls layers beginning from the top, ie. with the "most recent"
  /// layer. Return early when visitor returns something that evaluates to true.
  template <typename Visitor_t, typename... Args_t>
  auto visit_from_top(Visitor_t&& visitor, Args_t&&... args) const
  {
    decltype(visitor(rbegin(layers)->get(), std::forward<Args_t>(args)...)) res;
    if (res
        = visit(rbegin(layers), rend(layers), std::forward<Visitor_t>(visitor),
                std::forward<Args_t>(args)...);
        res)
      return res;
    if (check_other())
      res = other->get().get().visit_from_top(std::forward<Visitor_t>(visitor),
                                        std::forward<Args_t>(args)...);
    return res;
  }

  /// Access alls layers beginning from the bottom, ie. with the "oldest" layer.
  /// Return early when visitor returns true.
  template <typename Visitor_t, typename... Args_t>
  bool visit_from_bottom(Visitor_t&& visitor, Args_t&&... args) const
  {
    if (check_other())
      if (auto res = other->get().get().visit_from_bottom(
              std::forward<Visitor_t>(visitor), std::forward<Args_t>(args)...);
          res)
        return res;
    return visit(begin(layers), end(layers), std::forward<Visitor_t>(visitor),
                 std::forward<Args_t>(args)...);
  }

  /// Read the current state. This walks through all layers beginning with the
  /// most recent one. THis function returns when reader return something that
  /// evaluates to true or when there are no layers left to read.
  template <typename Reader_t, typename... Args_t>
  auto read(Reader_t&& reader, Args_t&&... args) const
  {
    return visit_from_top(std::forward<Reader_t>(reader),
                          std::forward<Args_t>(args)...);
  }

  /// Write to the current writing layer
  template <typename Writer_t, typename... Args_t>
  auto write(Writer_t&& writer, Args_t&&... args)
  {
    if (layers.empty())
      throw std::runtime_error("Cannot write to empty (zero layers) cache.");
    assert(check_layer(active_layer()));
    return writer(active_layer().get(), std::forward<Args_t>(args)...);
  }

private:
  /// Trigger cleanup of (removal of unsued) storage layers
  void layer_expired([[maybe_unused]] maybe_null<const Layer_t> expired)
  {
    layers.remove_if([](const weakly_owned<Layer_t>& layer) {
      return layer.use_count() == 0;
    });
  }

  /// Check whether a layer is safe to access
  bool check_layer(const weakly_owned<Layer_t>& layer) const
  {
    return check(layer);
  }
  bool check_other() const { return other.has_value() && check(*other); }

  /// The active layer
  weakly_owned<Layer_t>& active_layer()
  {
    assert(check(layers.back()));
    return layers.back();
  }

  // The background read-only storage
  std::optional<weakly_owned<std::reference_wrapper<const LayerManager>>> other{};

  // Own one layer. Enables remanent storage.
  std::optional<LayerLock_t> remanent_lock{};

  // The storage layers
  std::list<weakly_owned<Layer_t>> layers;

  // Lifetime information
  lock_t lifetime{std::make_shared<const bool>(true)};
};

} // namespace dolfinx::common::memory
