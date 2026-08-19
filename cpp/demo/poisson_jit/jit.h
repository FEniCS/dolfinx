// Copyright (C) 2026 Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <format>
#include <fstream>
#include <mpi.h>
#include <span>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <type_traits>
#include <ufcx.h>
#include <unistd.h>
#include <vector>

/// @brief Just-in-time compilation of UFL forms for the C++ interface.
///
/// A prototype of the mechanism used by the Python interface, expressed
/// entirely in terms of a C++ caller: UFL source held as a string is
/// passed to FFCx and a C compiler in a subprocess, and the resulting
/// shared library is loaded and searched for the `ufcx_form` symbols
/// that fem::create_form consumes.
///
/// POSIX only. FFCx-generated C carries no `__declspec(dllexport)`, so
/// nothing is exported from a DLL built by MSVC; Windows support needs
/// a generated module definition file.
namespace dolfinx_demo::jit
{
namespace impl
{
template <typename T>
inline constexpr bool always_false = false;

/// FNV-1a. Stable across toolchains, unlike std::hash, which matters
/// because the on-disk cache is shared between builds.
constexpr std::uint64_t hash(std::string_view s,
                             std::uint64_t h = 0xcbf29ce484222325)
{
  for (char c : s)
  {
    h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
    h *= 0x100000001b3;
  }
  return h;
}

/// Directory holding generated code, following the convention of
/// dolfinx.jit.
inline std::filesystem::path cache_dir()
{
  if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg and *xdg != '\0')
    return std::filesystem::path(xdg) / "fenics";
  const char* home = std::getenv("HOME");
  if (!home or *home == '\0')
    throw std::runtime_error("Neither XDG_CACHE_HOME nor HOME is set.");
  return std::filesystem::path(home) / ".cache" / "fenics";
}

/// Run a shell command, appending its output to @p log.
inline void run(const std::string& command, const std::filesystem::path& log)
{
  spdlog::debug("JIT: system() enter: {}", command);
  const std::string redirected
      = std::format("{} >> \"{}\" 2>&1", command, log.string());
  const int status = std::system(redirected.c_str());
  spdlog::debug("JIT: system() returned {}", status);
  if (status != 0)
  {
    // std::system returns a wait(2) status, not an exit code
    const int code = WIFEXITED(status) ? WEXITSTATUS(status) : status;
    throw std::runtime_error(std::format(
        "JIT command exited with status {}:\n  {}\nOutput was written to {}",
        code, command, log.string()));
  }
}

/// Generate C from @p ufl with FFCx and compile it into a shared library
/// at @p library. Not collective: called on one rank only.
inline void build(std::string_view ufl, std::string_view scalar_type,
                  const std::string& key, const std::filesystem::path& library)
{
  const std::filesystem::path cache = library.parent_path();
  const std::filesystem::path log = cache / std::format("{}.log", key);
  std::filesystem::remove(log);

  // Generate into a private directory and rename the finished library
  // into place, so that a concurrent job sharing the cache never
  // observes a partial build
  const std::filesystem::path tmp
      = cache / std::format("tmp-{}-{}", key, getpid());
  std::filesystem::create_directories(tmp);

  const std::filesystem::path source = tmp / std::format("{}.py", key);
  {
    std::ofstream file(source);
    if (!file)
      throw std::runtime_error(
          std::format("Could not write UFL source to {}", source.string()));
    file << ufl;
  }

  // `-n` fixes the prefix of the alias symbols FFCx emits for each named
  // UFL object, giving symbols that the caller can predict from `key`
  run(std::format(
          R"("{}" -m ffcx -i "{}" -n {} -o {} -d "{}" --scalar_type={})",
          JIT_PYTHON_EXECUTABLE, source.string(), key, key, tmp.string(),
          scalar_type),
      log);

  const std::filesystem::path object = tmp / std::format("{}.so", key);
  run(std::format(R"("{}" {} -I"{}" -o "{}" "{}" -lm)", JIT_C_COMPILER,
                  JIT_C_FLAGS, JIT_UFCX_INCLUDE_DIR, object.string(),
                  (tmp / std::format("{}.c", key)).string()),
      log);

  std::filesystem::rename(object, library);
  std::filesystem::remove_all(tmp);
}
} // namespace impl

/// @brief FFCx name for the scalar type `T`.
template <typename T>
consteval std::string_view scalar_type()
{
  if constexpr (std::is_same_v<T, float>)
    return "float32";
  else if constexpr (std::is_same_v<T, double>)
    return "float64";
  else if constexpr (std::is_same_v<T, std::complex<float>>)
    return "complex64";
  else if constexpr (std::is_same_v<T, std::complex<double>>)
    return "complex128";
  else
    static_assert(impl::always_false<T>, "Unsupported scalar type.");
}

/// @brief Compile UFL source and return the forms it defines.
///
/// Rank 0 of @p comm generates C with FFCx and compiles it to a shared
/// library in the cache directory; all ranks then load that library. A
/// library whose cache key already exists is reused without
/// recompilation.
///
/// @note The returned forms, and the integration kernels they point to,
/// are owned by a shared library that is never unloaded. fem::Form
/// stores the kernel pointers rather than copying the code, so unloading
/// would leave assembled forms holding dangling pointers.
///
/// @param[in] comm MPI communicator. Compilation is collective on
/// `comm`.
/// @param[in] ufl UFL source, as it would appear in a form file.
/// @param[in] names Names of the UFL variables to return, e.g. `{"a",
/// "L"}`.
/// @param[in] scalar_type FFCx scalar type, e.g. `"float64"`. See
/// ::scalar_type.
/// @return Pointer to a `ufcx_form` for each entry of @p names, in
/// order.
inline std::vector<ufcx_form*> compile_forms(MPI_Comm comm,
                                             std::string_view ufl,
                                             std::span<const std::string> names,
                                             std::string_view scalar_type)
{
  // Cache key covers everything that changes the generated code or the
  // way it is built
  std::uint64_t h = impl::hash(ufl);
  h = impl::hash(scalar_type, h);
  h = impl::hash(JIT_PYTHON_EXECUTABLE, h);
  h = impl::hash(JIT_FFCX_VERSION, h);
  h = impl::hash(JIT_C_COMPILER, h);
  h = impl::hash(JIT_C_FLAGS, h);
  const std::string key = std::format("f{:016x}", h);

  const std::filesystem::path cache = impl::cache_dir();
  std::filesystem::create_directories(cache);
  const std::filesystem::path library = cache / std::format("{}.so", key);

  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  // Compile on rank 0 and broadcast the outcome. All ranks reach every
  // collective below whether or not the build succeeded, then throw
  // together
  int status = 0;
  std::string error;
  spdlog::debug("JIT[{}]: enter, key {}", rank, key);
  if (rank == 0 and !std::filesystem::exists(library))
  {
    spdlog::info("JIT compiling UFL forms ({})", key);
    try
    {
      impl::build(ufl, scalar_type, key, library);
    }
    catch (const std::exception& e)
    {
      status = 1;
      error = e.what();
    }
  }
  spdlog::debug("JIT[{}]: build done, entering Bcast", rank);

  MPI_Bcast(&status, 1, MPI_INT, 0, comm);
  if (status != 0)
  {
    int size = error.size();
    MPI_Bcast(&size, 1, MPI_INT, 0, comm);
    error.resize(size);
    MPI_Bcast(error.data(), size, MPI_CHAR, 0, comm);
    throw std::runtime_error(
        std::format("JIT compilation failed on rank 0: {}", error));
  }

  // Rank 0 must have renamed the library into place before any other
  // rank opens it
  spdlog::debug("JIT[{}]: entering Barrier", rank);
  MPI_Barrier(comm);
  spdlog::debug("JIT[{}]: past Barrier, dlopen", rank);

  static std::vector<void*> handles;
  void* handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
  {
    throw std::runtime_error(std::format("Could not load JIT-compiled library "
                                         "{}: {}",
                                         library.string(), dlerror()));
  }
  handles.push_back(handle);

  std::vector<ufcx_form*> forms;
  forms.reserve(names.size());
  for (const std::string& name : names)
  {
    // FFCx emits `ufcx_form* form_<prefix>_<name>`, a pointer to the
    // form object, aliasing the signature-named object itself
    const std::string symbol = std::format("form_{}_{}", key, name);
    void* address = dlsym(handle, symbol.c_str());
    if (!address)
    {
      throw std::runtime_error(
          std::format("UFL source defines no form '{}' (symbol '{}' not found "
                      "in {})",
                      name, symbol, library.string()));
    }
    forms.push_back(*reinterpret_cast<ufcx_form**>(address));
  }

  spdlog::debug("JIT[{}]: resolved {} form(s)", rank, forms.size());
  return forms;
}
} // namespace dolfinx_demo::jit
