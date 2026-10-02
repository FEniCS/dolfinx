# Helpers for locating PETSc, SLEPc and packages installed alongside them
# using pkg-config.
#
# This module is installed next to DOLFINXConfig.cmake and included by it,
# so that a consumer of an installed DOLFINx discovers the same PETSc/SLEPc
# installation that DOLFINx was built against.

include_guard(GLOBAL)

# Prepend the pkgconfig directories of the prefix named by the environment
# variable ${dir_env_var} (e.g. PETSC_DIR) to PKG_CONFIG_PATH. Both
# <prefix>/lib/pkgconfig and, when PETSC_ARCH is set,
# <prefix>/$ENV{PETSC_ARCH}/lib/pkgconfig are added.
#
# A function rather than a macro: neither helper here sets a variable in
# the caller's scope, only the process environment, which a function
# changes just the same.
function(dolfinx_prepend_pkgconfig_path dir_env_var)
  if(NOT DEFINED ENV{${dir_env_var}})
    return()
  endif()

  if(WIN32)
    set(sep ";")
  else()
    set(sep ":")
  endif()

  # Read PKG_CONFIG_PATH through a guard rather than dereferencing it
  # directly: CI configures with --warn-uninitialized -Werror=dev, under
  # which reading an unset environment variable is an error.
  if(DEFINED ENV{PKG_CONFIG_PATH})
    set(pkg_config_path "$ENV{PKG_CONFIG_PATH}")
  else()
    set(pkg_config_path "")
  endif()

  set(
    ENV{PKG_CONFIG_PATH}
    "$ENV{${dir_env_var}}/lib/pkgconfig${sep}${pkg_config_path}"
  )
  if(DEFINED ENV{PETSC_ARCH})
    set(
      ENV{PKG_CONFIG_PATH}
      "$ENV{${dir_env_var}}/$ENV{PETSC_ARCH}/lib/pkgconfig${sep}$ENV{PKG_CONFIG_PATH}"
    )
  endif()
endfunction()

# Set the environment variable ${dir_env_var} to the prefix reported by the
# Python module ${module}, so that dolfinx_prepend_pkgconfig_path() finds
# the PETSc/SLEPc a Python installation was built against. A variable set
# by the user always wins.
function(dolfinx_python_prefix_hint module dir_env_var)
  if(DEFINED ENV{${dir_env_var}})
    return()
  endif()

  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(NOT Python3_Interpreter_FOUND)
    return()
  endif()

  execute_process(
    COMMAND
      ${Python3_EXECUTABLE} -c
      "import ${module}, sys; sys.stdout.write(${module}.get_${module}_dir())"
    OUTPUT_VARIABLE dir
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(dir)
    message(STATUS "Found ${module} Python module at ${dir}")
    set(ENV{${dir_env_var}} "${dir}")
  endif()
endfunction()
