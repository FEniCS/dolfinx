set(DOLFIN_UFC_FOUND 0)

include(FindPkgConfig)
pkg_check_modules(UFC REQUIRED ufc-1>=1.4.1)

if(UFC_FOUND)
  set(DOLFIN_UFC_INCLUDE_DIRS
    ${UFC_INCLUDE_DIRS}
    )

  if(DOLFIN_ENABLE_PYTHON)
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c "import sys, ufc; sys.stdout.write(ufc.__swigversion__)"
      OUTPUT_VARIABLE UFC_SWIGVERSION
      )
    if(NOT "${SWIG_VERSION}" STREQUAL "${UFC_SWIGVERSION}")
      message(FATAL_ERROR "UFC compiled with different version of SWIG. Please install SWIG version ${UFC_SWIGVERSION} or recompile UFC with present SWIG.")
    endif(NOT "${SWIG_VERSION}" STREQUAL "${UFC_SWIGVERSION}")
  endif(DOLFIN_ENABLE_PYTHON)

  set(DOLFIN_UFC_FOUND 1)
endif(UFC_FOUND)
