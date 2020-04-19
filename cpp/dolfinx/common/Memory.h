//
// Created by mrambausek on 18.04.20.
//

#pragma once

//
// Created by mrambausek on 18.04.20.
//

#pragma once

#include <cassert>
#include <memory>
#include <shared_mutex>
#include <vector>
#include <functional>


namespace dolfinx::common::memory
{

using lock_t = std::shared_ptr<const bool>;
using sentinel_t = std::weak_ptr<const bool>;


/// A wrapper for objects with lifetime dependencies without using smart
/// pointers but mimicking their interface to a certain degree.
template <typename T, typename observer_t>
class observed
{
public:
  observed(T&& obj, observer_t observer)
      : obj{std::move(obj)}, observer{std::move(observer)}
  {
    // do nothing
  }

  observed() = default;
  observed(const observed& other) = default;
  explicit observed(observed&& other) = default;

  observed& operator=(const observed& other) = default;
  observed& operator=(observed&& other) = default;

  ~observed() = default;

  T& get()
  {
    assert(use_count() != 0);
    return obj;
  }

  const T& get() const
  {
    assert(use_count() != 0);
    return obj;
  }

  [[nodiscard]] long use_count() const noexcept { return observer.use_count(); }

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

template <typename T>
weakly_owned<T> make_guarded(T obj, sentinel_t sentinel)
{
  return weakly_owned<T>{std::forward<T&&>(obj), std::move(sentinel)};
}

template <typename T>
bool check(const weakly_owned<T>& obj)
{
  return obj.use_count() != 0;
}

template <typename T>
bool check(const weakly_owned<T*>& obj)
{
  return obj.use_count() != 0 && obj.get();
}

template <typename T>
using owned = observed<T, lock_t>;

template <typename T>
owned<T> make_owned(T obj, lock_t lock)
{
  return owned<T>{std::forward<T&&>(obj), std::move(lock)};
}

/// Gives read-only or full access via visitor pattern of the underlying type
template <typename Accessed_t, typename... Locks_t>
class DataAccessor
{
public:
  explicit DataAccessor(Accessed_t* accessed, Locks_t... locks)
      : accessed{accessed}, locks{std::make_tuple(std::move(locks)...)}
  {
    // do nothing
  }

  DataAccessor() = delete;
  DataAccessor(const DataAccessor& other) = delete;
  DataAccessor(DataAccessor&& other) = default;
  DataAccessor& operator=(const DataAccessor& other) = delete;
  DataAccessor& operator=(DataAccessor&& other) = default;
  ~DataAccessor() = default;

  Accessed_t* operator->() noexcept { return accessed; }
  const Accessed_t* operator->() const noexcept { return accessed; }

private:
  Accessed_t* accessed;
  std::tuple<Locks_t...> locks;
};

template <typename Get_Read_t, typename Get_Read_Write_t, typename... Locks_t>
class SafeHandle
{
public:
  explicit SafeHandle(Get_Read_t get_read, Get_Read_Write_t get_read_write,
                      Locks_t... locks)
      : get_read{std::move(get_read)},
        get_read_write{std::move(get_read_write)}, locks{std::make_tuple(
          std::move(locks)...)} {
    // do nothing
  };

  auto read_access() const { return get_read(); }
  auto read_write_access() { return get_read_write(); }

private:
  Get_Read_t get_read{};
  Get_Read_Write_t get_read_write{};
  std::tuple<Locks_t...> locks;
};

template <typename Iterator_t, typename Visitor_t>
auto visit(Iterator_t begin, Iterator_t end, Visitor_t&& visitor)
{
  decltype(visitor(begin->get())) res;
  for (auto layer = begin; layer != end; ++layer)
  {
    assert(check(*layer));
    if (res = visitor(layer->get()); res)
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
  LayerLock(owned<const Layer_t&> layer,
            weakly_owned<std::function<void()>> on_destruction)
      : _layer{std::move(layer)}, _on_destruction{std::move(on_destruction)}
  {
    // do nothing
  }

  LayerLock(const LayerLock&) = default;
  LayerLock(LayerLock&&) = default;

  LayerLock& operator=(const LayerLock&) = default;
  LayerLock& operator=(LayerLock&&) = default;

  ~LayerLock() { release();};

  /// Returns a pointer to the owned layer. If no layer os owned because
  /// the lock has been released.
  maybe_null<const Layer_t> layer() const
  {
    return check(_layer) ? &_layer : nullptr;
  }

  /// Relases ownership.
  void release()
  {
    _layer.reset();
    // Signalize to the layer manager that a lock has gone
    if (check(_on_destruction))
      _on_destruction.get()(layer());
  };

private:
  // Owned layer
  owned<const Layer_t&> _layer;

  // Callback informing that the lock has been released
  weakly_owned<std::function<void()>> _on_destruction;
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
  using Lock_t = LayerLock<Layer_t>;

  // TODO: [MULTITHREADING] remove if not needed
  using ReadPtr
  = DataAccessor<const LayerManager,
      std::vector<std::shared_lock<std::shared_mutex>>>;
  using ReadWritePtr
  = DataAccessor<LayerManager, std::unique_lock<std::shared_mutex>,
      std::vector<std::shared_lock<std::shared_mutex>>>;

  /// Create Storage layer
  /// @param[in] remanent if given creates a remanent storage layer of which the
  /// lifetime is bound to this object.
  /// @param[in] other points to another LayerManager for read only access.
  /// Its data will not be copied into the new LayerManager.
  LayerManager(bool remanent, maybe_null<const LayerManager> other)
      : other{make_guarded(other, other->lifetime)}
  {
    if (remanent)
      remanent_lock = hold_layer(true);
  }

  LayerManager(bool remanent) : other{make_guarded(nullptr, sentinel_t{})}
  {
    if (remanent)
      remanent_lock = hold_layer(true);
  }

  LayerManager() = default;

  /// Copy constructor [deleted]
  LayerManager(const LayerManager& other) = delete;

  /// Move constructor
  explicit LayerManager(LayerManager&& other) = default;

  /// Destructor
  ~LayerManager() = default;

  /// Copy assignment [deleted]
  LayerManager& operator=(const LayerManager& other) = delete;

  /// Move assignment
  LayerManager& operator=(LayerManager&& other) = default;

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
  /// [remove if not needed]
  /// Thread safety: it is only frozen during "safe access".
  Lock_t hold_layer(bool force_new_layer = false)
  {
    auto layer_lock = std::make_shared<const bool>(true);
    if (layers.empty() or force_new_layer)
    {
      layers.emplace_back(Layer_t{}, layer_lock);
    }
    return Lock_t(make_owned<const Layer_t&>(&active_layer().get(),
                                             std::move(layer_lock)),
                  make_guarded([=]() { this->layer_expired(); }, lifetime));
  }

  void push_layer(Layer_t layer) { layers.emplace_back(std::move(layer)); }

  bool is_remanent() { return remanent_lock.has_value(); }

  bool empty() const { return layers.empty(); }

  bool has_other() const { return check_other(); }

  maybe_null<const LayerManager> get_other() const { return other; }

  template <typename Visitor_t>
  auto visit_from_top(Visitor_t&& visitor) const
  {
    decltype(visitor(rbegin(layers)->get())) res;
    if (res
            = visit(rbegin(layers), rend(layers), std::forward<Visitor_t>(visitor));
        res)
      return res;
    if (check_other())
      res = other.get()->visit_from_top(std::forward<Visitor_t>(visitor));
    return res;
  }

  template <typename Visitor_t>
  auto visit_from_bottom(Visitor_t&& visitor) const
  {
    if (check_other())
      if (auto res = other.get()->visit_from_bottom(std::forward<Visitor_t>(visitor));
          res)
        return res;
    return visit(begin(layers), end(layers), std::forward<Visitor_t>(visitor));
  }

  template <typename Reader_t>
  auto read(Reader_t&& reader) const
  {
    return visit_from_top(std::forward<Reader_t>(reader));
  }

  template <typename Writer, typename... Args_t>
  auto write(Writer&& writer)
  {
    if (layers.empty())
      throw std::runtime_error("Cannot write to empty (zero layers) cache.");
    assert(check_layer(active_layer()));
    return writer(active_layer().get());
  }

  // TODO: [MULTITHREADING] remove if not needed
  auto safe_handle()
  {
    // Protect the the layer lock! Do not ask for a lock if there is nothing to
    // lock. New layers have to be created by accessors
    auto lock = read_lock();
    auto layer_lock = !empty() ? hold_layer() : Lock_t{};
    return SafeHandle{[=]() { safe_read_access(); },
                      [=]() { safe_read_write_access(); },
                      std::move(layer_lock),
                      check_other() ? other.get()->read_locks()
                                    : decltype(other.get()->read_locks()){}};
  }

private:
  void layer_expired()
  {
    if (check_layer(layers.back()))
      return;

    // TODO: [MULTITHREADING] remove if not needed
    auto write_lock = std::unique_lock(mtx);

    while (!layers.empty() && !check_layer(layers.back()))
      layers.pop_back();
  }

  bool check_layer(const weakly_owned<Layer_t>& layer) const { return check(layer); }
  bool check_other() const { return check(other); }

  /// The active layer
  weakly_owned<Layer_t>& active_layer()
  {
    assert(check(layers.back()));
    return layers.back();
  }

  weakly_owned<maybe_null<const LayerManager>> other;

  // Own one layer. Is this useful? For certain cases yes.
  std::optional<Lock_t> remanent_lock{};

  // The storage layers
  std::vector<weakly_owned<Layer_t>> layers;

  // Lifetime information
  lock_t lifetime{std::make_shared<const bool>(true)};

  // TODO: remove if not needed
  // Protecting read/write
  mutable std::shared_mutex mtx;

  std::shared_lock<std::shared_mutex> read_lock() const
  {
    return std::shared_lock(mtx);
  }

  std::vector<std::shared_lock<std::shared_mutex>> read_locks() const
  {
    std::vector<std::shared_lock<std::shared_mutex>> locks;
    if (check_other())
      other.first->stack_locks(locks);
    stack_locks(locks);
    return locks;
  };

  void
  stack_locks(std::vector<std::shared_lock<std::shared_mutex>>& locks) const
  {
    locks.push_back(read_lock());
  }

  std::unique_lock<std::shared_mutex> read_write_lock() const
  {
    return std::unique_lock(mtx);
  }

  ReadPtr safe_read_access() const { return ReadPtr{this, read_lock()}; }

  ReadWritePtr safe_read_write_access()
  {
    return ReadWritePtr{this, read_write_lock()};
  }
};

} // namespace dolfinx::common::memory
