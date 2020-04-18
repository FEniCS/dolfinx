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

namespace dolfinx::common::memory
{

using lock_t = std::shared_ptr<const bool>;
using sentinel_t = std::weak_ptr<const bool>;

template <typename T>
using guarded_obj = std::pair<T, sentinel_t>;

template <typename T>
guarded_obj<T> make_guarded(T obj, sentinel_t sentinel)
{
  return std::make_pair<typename guarded_obj<T>::first_type,
                        typename guarded_obj<T>::second_type>(
      std::forward<T&&>(obj), std::move(sentinel));
}

template <typename T>
bool check(const guarded_obj<T>& obj)
{
  return !obj.second.expired();
}

template <typename T>
bool check(const guarded_obj<T*>& obj)
{
  return !obj.second.expired() && obj;
}

template <typename T>
using owned_obj = std::pair<T, lock_t>;

template <typename T>
owned_obj<T> make_owned(T obj, lock_t lock)
{
  return std::make_pair<typename guarded_obj<T>::first_type,
                        typename owned_obj<T>::second_type>(
      std::forward<T&&>(obj), std::move(lock));
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

template <typename Iterator_t, typename Visitor_t, typename... Args_t>
auto visit(Iterator_t begin, Iterator_t end, Visitor_t&& visitor,
           Args_t&&... args)
{
  decltype(visitor(begin->first, std::forward<Args_t>(args)...)) res;
  for (auto layer = begin; layer != end; ++layer)
  {
    assert(check(*layer));
    if (res = visitor(layer->first, std::forward<Args_t>(args)...); res)
      return res;
  }
  return res;
}

template <typename Layer_t>
class LayerManager;

template <typename Layer_t>
class LayerLock
{
public:
  // TODO: The ptr to the active layer is not be necessary ATM but reminds
  // of considering using shared/weak_ptr for the layers.
  LayerLock(owned_obj<const Layer_t*> layer,
            guarded_obj<std::function<void()>> on_destruction)
      : _layer{std::move(layer)}, _on_destruction{std::move(on_destruction)}
  {
    // do nothing
  }

  LayerLock() = default;
  LayerLock(const LayerLock&) = delete;
  LayerLock(LayerLock&&) = default;

  LayerLock& operator=(const LayerLock&) = delete;
  LayerLock& operator=(LayerLock&&) = default;

  ~LayerLock() { release(); };

  const Layer_t* layer() const { return _layer; }

private:
  void release()
  {
    _layer.second.reset();
    // Signalize to the layer manager that a lock has gone
    if (check(_on_destruction))
      _on_destruction.first();
  };

private:
  owned_obj<const Layer_t*> _layer;
  guarded_obj<std::function<void()>> _on_destruction;
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

// What if locks do not call back? No call back, just clean upon each access as
// in initial designs. When considering a stack-like memory
// (intermediate layers are not to be removed) this should not really be an
// issue. Possibly do both, also fixing issue of copying (losing locks).
// Thus, it would be nice to not have to store a layer manager as ptr.
//
// Since guarded_obj holds objects, constness is respected here. But also means
// that memory is not automatically freed, its the "guard" who loses its
// shared_ptr.
// V1: shared_ptr -> no automatric free of memory, remove if use count = 1
// this is quite equivalent to the guarded_obj model
// V2: weak_ptr -> memory immediatly freed, only layer stack is cleaned up
// V3: shared_ptr/guarded with callback and removal from within the stack (no
// stack model anymore).
//  '-> Concerning copy: Only allowed for weak_ptr model.
// It is a very bad idea to store the layer manager as a (shared) ptr since
// one does not want to have it constant, but then two object lose same data.
// which means, one is just a view, not a copy.


template <typename Layer_t>
class LayerManager
{
public:
  using Lock_t = LayerLock<Layer_t>;

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
  LayerManager(bool remanent, const LayerManager* other)
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

  // TODO: the copy constructor should not be default. While this basically
  // works, the locks from "other" will not trigger cleanup on "this".
  // Note that the guarded object still know whether they are expired, cleanup
  // is just not triggered. So it is better to forbid copying and store either
  // in a shared_ptr or implement a proper copy constructor.
  /// Copy constructor.
  LayerManager(const LayerManager& other) = default;

  /// Move constructor
  LayerManager(LayerManager&& other) = default;

  /// Destructor
  ~LayerManager() = default;

  /// Copy assignment
  LayerManager& operator=(const LayerManager& other) = default;

  /// Move assignment
  LayerManager& operator=(LayerManager&& other) = default;

  // TODO: terminology
  /// Create a new lock-like handle to keep a storage layer alive. By default,
  /// this returns just is another handle for the current layer. The creation of
  /// a new write layer is optional. Note that layers will be destroyed in
  /// reverse order of creation. This means, they do not immediately vanish if
  /// the locks leave their scope [easy to change, depends on whether this is a
  /// valid use case at all].
  /// Once a new layer is active, lower layers cannot be written to. This
  /// creates the possibility to recover the previous state. Also not that
  /// while lower layer cannot be written to, they can be read.
  /// We remark that the read-only "other" data may still be changed from
  /// outside. It is only frozen during "safe access".
  Lock_t hold_layer(bool force_new_layer=false)
  {
    auto layer_lock = std::make_shared<const bool>(true);
    if (layers.empty() or force_new_layer)
    {
      layers.emplace_back(Layer_t{}, layer_lock);
    }
    return Lock_t(make_owned<const Layer_t*>(&active_layer().first,
                                             std::move(layer_lock)),
                  make_guarded([=]() { this->layer_expired(); }, lifetime));
  }

  void push_layer(Layer_t layer) { layers.emplace_back(std::move(layer)); }

  bool is_remanent() { return remanent_lock.first != nullptr; }

  bool empty() const { return layers.empty(); }

  template <typename Visitor_t, typename... Args_t>
  auto visit_from_top(Visitor_t&& visitor, Args_t&&... args) const
  {
    decltype(visitor(rbegin(layers)->first, std::forward<Args_t>(args)...)) res;
    if (res
        = visit(rbegin(layers), rend(layers), std::forward<Visitor_t>(visitor),
                std::forward<Args_t>(args)...);
        res)
      return res;
    if (check_other())
      res = other.first->visit_from_top(std::forward<Visitor_t>(visitor),
                                        std::forward<Args_t>(args)...);
    return res;
  }

  template <typename Visitor_t, typename... Args_t>
  auto visit_from_bottom(Visitor_t&& visitor, Args_t&&... args) const
  {
    if (check_other())
      if (auto res = other.first->visit_from_bottom(
              std::forward<Visitor_t>(visitor), std::forward<Args_t>(args)...);
          res)
        return res;
    return visit(begin(layers), end(layers), std::forward<Visitor_t>(visitor),
                 std::forward<Args_t>(args)...);
  }

  template <typename Reader_t, typename... Args_t>
  auto read(Reader_t&& reader, Args_t&&... args) const
  {
    return visit_from_top(std::forward<Reader_t>(reader),
                          std::forward<Args_t>(args)...);
  }

  template <typename Writer, typename... Args_t>
  auto write(Writer&& writer, Args_t&&... args)
  {
    if (layers.empty())
      throw std::runtime_error("Cannot write to empty (zero layers) cache.");
    auto& active = active_layer();
    assert(!active.second.expired());
    return writer(active.first, std::forward<Args_t>(args)...);
  }

  auto safe_handle()
  {
    // Protect the the layer lock! Do not ask for a lock if there is nothing to
    // lock. New layers have to be created by accessors
    auto lock = read_lock();
    auto layer_lock = !empty() ? hold_layer() : Lock_t{};
    return SafeHandle{[=]() { safe_read_access(); },
                      [=]() { safe_read_write_access(); }, std::move(layer_lock),
                      check_other() ? other.first->read_locks()
                                    : decltype(other.first->read_locks()){}};
  }

private:
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

  ReadPtr safe_read_access() const
  {
    return ReadPtr{this, read_lock()};
  }

  ReadWritePtr safe_read_write_access()
  {
    return ReadWritePtr{this, read_write_lock()};
  }

  void layer_expired()
  {
    if (!layers.back().second.expired())
      return;

    auto write_lock = std::unique_lock(mtx);
    while (!layers.empty() && layers.back().second.expired())
      layers.pop_back();
  }

  bool check_layer(const Layer_t& layer) const { return check(layer); }

  bool check_other() const { return !other.second.expired() && other.first; }

  /// The active layer
  guarded_obj<Layer_t>& active_layer()
  {
    assert(!layers.back().second.expired());
    return layers.back();
  }

  guarded_obj<const LayerManager*> other;

  // Own one layer. Is this useful? For certain cases yes.
  Lock_t remanent_lock{};

  // The storage layers
  std::vector<guarded_obj<Layer_t>> layers;

  // Lifetime information
  lock_t lifetime{std::make_shared<const bool>(true)};
};

} // namespace dolfinx::common::memory
