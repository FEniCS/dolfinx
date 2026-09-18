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
macro(dolfinx_prepend_pkgconfig_path dir_env_var)
  block(SCOPE_FOR VARIABLES)
    if(WIN32)
      set(_sep ";")
    else()
      set(_sep ":")
    endif()
    if(DEFINED ENV{PKG_CONFIG_PATH})
      set(_orig "$ENV{PKG_CONFIG_PATH}")
    else()
      set(_orig "")
    endif()
    if(DEFINED ENV{${dir_env_var}})
      set(
        ENV{PKG_CONFIG_PATH}
        "$ENV{${dir_env_var}}/lib/pkgconfig${_sep}${_orig}"
      )
      if(DEFINED ENV{PETSC_ARCH})
        set(
          ENV{PKG_CONFIG_PATH}
          "$ENV{${dir_env_var}}/$ENV{PETSC_ARCH}/lib/pkgconfig${_sep}$ENV{PKG_CONFIG_PATH}"
        )
      endif()
    endif()
  endblock()
endmacro()

# Set the environment variable ${dir_env_var} to the prefix reported by the
# Python module ${module}, so that dolfinx_prepend_pkgconfig_path() finds
# the PETSc/SLEPc a Python installation was built against. A variable set
# by the user always wins.
macro(dolfinx_python_prefix_hint module dir_env_var)
  block(SCOPE_FOR VARIABLES)
    if(NOT DEFINED ENV{${dir_env_var}})
      find_package(Python3 COMPONENTS Interpreter QUIET)
      if(Python3_Interpreter_FOUND)
        execute_process(
          COMMAND
            ${Python3_EXECUTABLE} -c
            "import ${module}, sys; sys.stdout.write(${module}.get_${module}_dir())"
          OUTPUT_VARIABLE _dir
          ERROR_QUIET
          OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(_dir)
          message(STATUS "Found ${module} Python module at ${_dir}")
          set(ENV{${dir_env_var}} "${_dir}")
        endif()
      endif()
    endif()
  endblock()
endmacro()
