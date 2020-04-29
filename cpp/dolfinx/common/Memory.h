// Copyright (C) 2020 Matthias Rambausek
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cassert>
#include <functional>
#include <list>
#include <memory>
#include <shared_mutex>
#include <vector>

namespace dolfinx::common::memory
{

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
  LayerLock(std::shared_ptr<const Layer_t> layer) : _layer{std::move(layer)}
  {
    // do nothing
  }

  /// Relases ownership.
  void release() { _layer.reset(); };

private:
  // Owned layer
  std::shared_ptr<const Layer_t> _layer;
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
  /// Its data will not be copied into the new LayerManager but the reference to
  /// other will be stored. The user is responsible for ensuring that the
  /// pointer to other is not dangling.
  LayerManager(bool remanent, const LayerManager* other) : other{other}
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

  // Copy constructor [deleted]
  LayerManager(const LayerManager& other) = delete;

  /// Move constructor
  explicit LayerManager(LayerManager&& other) noexcept = default;

  /// Destructor
  ~LayerManager() = default;

  // Copy assignment [deleted]
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
    layer_expired();
    if (layers.empty() or force_new_layer)
    {
      auto locked_layer = std::make_shared<Layer_t>();
      layers.emplace_back(locked_layer);
      return LayerLock_t{locked_layer};
    }
    else
    {
      assert(active_layer() != nullptr);
      return LayerLock_t{active_layer()};
    }
  }

  /// Add a whole new layer from outside
  void push_layer(Layer_t layer) { layers.emplace_back(std::move(layer)); }

  /// Ask whether the storage has a remanent layer
  bool is_remanent() { return remanent_lock.has_value(); }

  /// Ask whether the storage is empty, i.e. does not have any writable layers.
  bool empty() const { return std::find_if(begin(layers), end(layers), check_layer) == end(layers); }

  /// Ask whether the storage has read-only background storage attached.
  bool has_other() const { return check_other(); }

  /// Get the attached read-only storage
  const LayerManager& get_other() const { return *other; }

  /// Read the current state. This walks through all layers beginning with the
  /// most recent one. This function returns when reader return something that
  /// evaluates to true or when there are no layers left to read.
  template <typename Reader_t, typename... Args_t,
            typename res_t = decltype(std::declval<Reader_t>()(
                std::declval<const Layer_t&>(), std::declval<Args_t>()...))>
  res_t read(Reader_t&& reader, Args_t&&... args) const
  {
    for (auto layer_it = rbegin(layers); layer_it != rend(layers); ++layer_it)
    {
      if (auto layer = extract_layer(layer_it); layer)
        if (auto res = reader(*layer, std::forward<Args_t>(args)...); res)
          return res;
    }
    if (check_other())
      return other->read(std::forward<Reader_t>(reader),
                         std::forward<Args_t>(args)...);

    return res_t{};
  }

  /// Read the current state begginning from the bottom layer.
  /// This function returns when reader return something that evaluates to true
  /// or when there are no layers left to read.
  template <typename Reader_t, typename... Args_t,
            typename res_t = decltype(std::declval<Reader_t>()(
                std::declval<const Layer_t&>(), std::declval<Args_t>()...))>
  res_t read_from_bottom(Reader_t&& reader, Args_t&&... args) const
  {
    res_t res{};
    if (check_other())
      if (res = other->read_from_bottom(std::forward<Reader_t>(reader),
                                        std::forward<Args_t>(args)...);
          res)
        return res;

    for (auto layer_it = begin(layers); layer_it != end(layers); ++layer_it)
    {
      if (auto layer = extract_layer(layer_it); layer)
        if (res = reader(*layer, std::forward<Args_t>(args)...); res)
          return res;
    }
    return res_t{};
  }

  /// Write to the current writing layer
  template <typename Writer_t, typename... Args_t>
  auto write(Writer_t&& writer, Args_t&&... args)
  {
    layer_expired();
    if (auto layer = active_layer(); layer)
      return writer(*layer, std::forward<Args_t>(args)...);
    else
      throw std::runtime_error("There is no layer for writing.");
  }

private:
  // Cleanup of (removal of unsued) storage layers
  void layer_expired()
  {
    layers.remove_if([](const std::weak_ptr<Layer_t>& layer) {
      return !LayerManager::check_layer(layer);
    });
  }

  // extract an accessible (shared) pointer from an iterator over layers
  template <typename It>
  std::shared_ptr<Layer_t> extract_layer(It layer_it)
  {
    if (auto weak_layer = *layer_it; check_layer(weak_layer))
      return weak_layer.lock();
    else
      return {nullptr};
  }

  // extract an accessible (shared) pointer from an iterator over layers (const)
  template <typename It>
  std::shared_ptr<const Layer_t> extract_layer(It layer_it) const
  {
    if (auto weak_layer = *layer_it; check_layer(weak_layer))
      return weak_layer.lock();
    else
      return {nullptr};
  }

  // Check whether a layer is safe to access
  static bool check_layer(const std::weak_ptr<Layer_t>& layer)
  {
    return !layer.expired() && layer.lock() != nullptr;
  }

  bool check_other() const { return other != nullptr; }

  // The active layer
  std::shared_ptr<Layer_t> active_layer()
  {
    if (!layers.empty() && check_layer(layers.back()))
      return layers.back().lock();
    else
      return {nullptr};
  }

  // The active layer (const)
  std::shared_ptr<const Layer_t> active_layer() const
  {
    if (!layers.empty() && check_layer(layers.back()))
      return layers.back().lock();
    else
      return {nullptr};
  }

  // The background read-only storage
  const LayerManager* other{nullptr};

  // Own one layer. Enables remanent storage.
  std::optional<LayerLock_t> remanent_lock{};

  // The storage layers
  std::list<std::weak_ptr<Layer_t>> layers;
};

} // namespace dolfinx::common::memory
