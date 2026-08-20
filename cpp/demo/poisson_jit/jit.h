// Copyright (C) 2026 Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef JIT_EMBED_PYTHON
// Python.h sets feature-test macros and must precede the standard headers
#include <Python.h>
#endif

#include <boost/asio/io_context.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/readable_pipe.hpp>
#include <boost/dll/shared_library.hpp>
#include <boost/dll/shared_library_load_mode.hpp>
#include <boost/process.hpp>
#include <boost/system/error_code.hpp>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <mpi.h>
#include <ranges>
#include <span>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <ufcx.h>
#include <vector>

/// @brief Just-in-time compilation of UFL forms for the C++ interface.
///
/// A prototype of the mechanism used by the Python interface, expressed
/// entirely in terms of a C++ caller: UFL source held as a string is
/// passed to FFCx and a C compiler with Boost.Process, and the resulting
/// shared library is loaded with Boost.DLL and searched for the
/// `ufcx_form` symbols that fem::create_form consumes.
///
/// Both libraries are portable, but the demo is restricted to POSIX
/// because FFCx-generated C carries no `__declspec(dllexport)`, so
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

/// Split a space-separated list of compiler flags into arguments. The
/// input is a build-time constant, so no quoting or escaping is
/// supported.
inline std::vector<std::string> split(std::string_view flags)
{
  std::vector<std::string> args;
  for (const auto& arg : std::views::split(flags, ' '))
    if (!std::ranges::empty(arg))
      args.emplace_back(std::ranges::begin(arg), std::ranges::end(arg));
  return args;
}

/// @brief Run @p exe with @p args, capturing its output.
///
/// No shell is involved, so arguments need no quoting and paths
/// containing spaces or shell metacharacters are passed through
/// unaltered.
///
/// @throws std::runtime_error if the process exits non-zero, with its
/// captured output included in the message.
inline void run(const std::filesystem::path& exe,
                const std::vector<std::string>& args)
{
  namespace bp = boost::process;
  std::string display = exe.string();
  for (const std::string& arg : args)
    display += " " + arg;
  spdlog::debug("JIT: running {}", display);

  // stdout and stderr need separate pipes: binding one pipe to both
  // replaces the first binding
  boost::asio::io_context ctx;
  boost::asio::readable_pipe out(ctx);
  boost::asio::readable_pipe err(ctx);
  bp::process proc(ctx, exe, args, bp::process_stdio{{}, out, err});

  // Drain both pipes before waiting, or a child that fills one blocks.
  // The reads are asynchronous so that they go through the asio reactor,
  // which resumes on EINTR; a blocking read fails when a signal arrives.
  std::string output, errors;
  boost::system::error_code out_ec, err_ec;
  boost::asio::async_read(out, boost::asio::dynamic_buffer(output),
                          [&out_ec](const boost::system::error_code& e,
                                    std::size_t) { out_ec = e; });
  boost::asio::async_read(err, boost::asio::dynamic_buffer(errors),
                          [&err_ec](const boost::system::error_code& e,
                                    std::size_t) { err_ec = e; });
  ctx.run();
  for (const boost::system::error_code& ec : {out_ec, err_ec})
  {
    if (ec and ec != boost::asio::error::eof)
    {
      throw std::runtime_error(std::format("Could not read output of {}: {}",
                                           exe.string(), ec.message()));
    }
  }

  const int code = proc.wait();
  spdlog::debug("JIT: {} exited with code {}", exe.string(), code);
  if (code != 0)
  {
    throw std::runtime_error(std::format("{} exited with code {}:\n{}{}",
                                         exe.string(), code, output, errors));
  }
}

#ifdef JIT_EMBED_PYTHON
/// Format the active Python exception the way the interpreter would
/// print it. The GIL must be held.
inline std::string python_error()
{
  PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
  PyErr_Fetch(&type, &value, &traceback);
  PyErr_NormalizeException(&type, &value, &traceback);

  std::string message = "Python raised an exception, with no traceback.";
  if (PyObject* module = PyImport_ImportModule("traceback"))
  {
    PyObject* lines = PyObject_CallMethod(
        module, "format_exception", "OOO", type ? type : Py_None,
        value ? value : Py_None, traceback ? traceback : Py_None);
    if (lines)
    {
      PyObject* separator = PyUnicode_FromString("");
      if (PyObject* joined = PyUnicode_Join(separator, lines))
      {
        message = PyUnicode_AsUTF8(joined);
        Py_DECREF(joined);
      }
      Py_XDECREF(separator);
      Py_DECREF(lines);
    }
    else
      PyErr_Clear();
    Py_DECREF(module);
  }
  else
    PyErr_Clear();

  Py_XDECREF(type);
  Py_XDECREF(value);
  Py_XDECREF(traceback);
  return message;
}

/// @brief Start the interpreter, once per process.
///
/// Never finalised: finalisation would have to be ordered against
/// MPI_Finalize and PetscFinalize, and the process is about to exit. An
/// interpreter that is already running, as when this code is reached
/// from the Python interface, is adopted rather than replaced.
inline void start_interpreter()
{
  static const bool started = []
  {
    if (Py_IsInitialized())
      return true;

    PyConfig config;
    PyConfig_InitPythonConfig(&config);

    // Naming the interpreter that owns FFCx is what makes its virtual
    // environment, and so its site-packages, visible here
    PyStatus status = PyConfig_SetBytesString(&config, &config.program_name,
                                              JIT_PYTHON_EXECUTABLE);
    if (!PyStatus_Exception(status))
      status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);

    if (PyStatus_Exception(status))
    {
      throw std::runtime_error(
          std::format("Could not start the Python interpreter ({}): {}",
                      JIT_PYTHON_EXECUTABLE,
                      status.err_msg ? status.err_msg : "unknown error"));
    }
    return true;
  }();
  (void)started;
}

/// @brief Run FFCx in this process, with the arguments its command line
/// takes.
///
/// @throws std::runtime_error carrying the Python traceback.
inline void run_ffcx(const std::vector<std::string>& args)
{
  start_interpreter();

  std::string display;
  for (const std::string& arg : args)
    display += arg + " ";
  spdlog::debug("JIT: ffcx {}", display);

  // Collect any error while the GIL is held, and throw once it is not
  std::string error;
  const PyGILState_STATE gil = PyGILState_Ensure();
  if (PyObject* module = PyImport_ImportModule("ffcx.main"))
  {
    if (PyObject* main = PyObject_GetAttrString(module, "main"))
    {
      PyObject* argv = PyList_New(0);
      for (const std::string& arg : args)
      {
        PyObject* item = PyUnicode_FromString(arg.c_str());
        PyList_Append(argv, item);
        Py_DECREF(item);
      }

      PyObject* result = PyObject_CallOneArg(main, argv);
      if (!result)
        error = python_error();
      else
      {
        if (long code = PyLong_AsLong(result); code != 0)
          error = std::format("FFCx returned {}.", code);
        Py_DECREF(result);
      }

      Py_DECREF(argv);
      Py_DECREF(main);
    }
    else
      error = python_error();
    Py_DECREF(module);
  }
  else
    error = python_error();
  PyGILState_Release(gil);

  if (!error.empty())
    throw std::runtime_error(std::format("FFCx failed:\n{}", error));
}
#endif

/// Generate C from @p ufl with FFCx and compile it into a shared library
/// at @p library. Not collective: called on one rank only.
inline void build(std::string_view ufl, std::string_view scalar_type,
                  const std::string& key, const std::filesystem::path& library)
{
  const std::filesystem::path cache = library.parent_path();

  // Generate into a private directory and rename the finished library
  // into place, so that a concurrent job sharing the cache never
  // observes a partial build
  const std::filesystem::path tmp
      = cache / std::format("tmp-{}-{}", key, boost::process::current_pid());
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
  const std::vector<std::string> ffcx_args
      = {"-i",
         source.string(),
         "-n",
         key,
         "-o",
         key,
         "-d",
         tmp.string(),
         std::format("--scalar_type={}", scalar_type)};
#ifdef JIT_EMBED_PYTHON
  run_ffcx(ffcx_args);
#else
  std::vector<std::string> argv = {"-m", "ffcx"};
  argv.insert(argv.end(), ffcx_args.begin(), ffcx_args.end());
  run(JIT_PYTHON_EXECUTABLE, argv);
#endif

  const std::filesystem::path object = tmp / std::format("{}.so", key);
  std::vector<std::string> cc_args = split(JIT_C_FLAGS);
  cc_args.insert(cc_args.end(),
                 {std::format("-I{}", JIT_UFCX_INCLUDE_DIR), "-o",
                  object.string(), (tmp / std::format("{}.c", key)).string(),
                  "-lm"});
  run(JIT_C_COMPILER, cc_args);

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
  spdlog::debug("JIT[{}]: past Barrier, loading library", rank);

  // Never unloaded: fem::Form stores the kernel pointers rather than
  // copying the code
  static std::vector<boost::dll::shared_library> libraries;
  libraries.emplace_back(library.string(),
                         boost::dll::load_mode::rtld_now
                             | boost::dll::load_mode::rtld_local);
  const boost::dll::shared_library& lib = libraries.back();

  std::vector<ufcx_form*> forms;
  forms.reserve(names.size());
  for (const std::string& name : names)
  {
    // FFCx emits `ufcx_form* form_<prefix>_<name>`, a pointer to the
    // form object, aliasing the signature-named object itself
    const std::string symbol = std::format("form_{}_{}", key, name);
    if (!lib.has(symbol))
    {
      throw std::runtime_error(
          std::format("UFL source defines no form '{}' (symbol '{}' not found "
                      "in {})",
                      name, symbol, library.string()));
    }
    forms.push_back(lib.get<ufcx_form*>(symbol));
  }

  spdlog::debug("JIT[{}]: resolved {} form(s)", rank, forms.size());
  return forms;
}
} // namespace dolfinx_demo::jit
