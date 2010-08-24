# - Find NumPy
# Find the native NumPy includes
# This module defines
#  NUMPY_INCLUDE_DIR, where to find numpy/arrayobject.h, etc.
#  NUMPY_FOUND, If false, do not try to use NumPy headers.

if(NUMPY_INCLUDE_DIR)
  # in cache already
  set(NUMPY_FIND_QUIETLY TRUE)
endif(NUMPY_INCLUDE_DIR)

exec_program("${PYTHON_EXECUTABLE}"
  ARGS "-c 'import numpy; print numpy.get_include()'"
  OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
  RETURN_VALUE NUMPY_NOT_FOUND)

if(NUMPY_INCLUDE_DIR)
  set(NUMPY_FOUND TRUE)
  set(NUMPY_INCLUDE_DIR ${NUMPY_INCLUDE_DIR} CACHE STRING "NumPy include path")
else(NUMPY_INCLUDE_DIR)
  set(NUMPY_FOUND FALSE)
endif(NUMPY_INCLUDE_DIR)

if(NUMPY_FOUND)
  if(NOT NUMPY_FIND_QUIETLY)
    message(STATUS "NumPy headers found")
  endif(NOT NUMPY_FIND_QUIETLY)
else(NUMPY_FOUND)
  if(NUMPY_FIND_REQUIRED)
    message(FATAL_ERROR "NumPy headers missing")
  endif(NUMPY_FIND_REQUIRED)
endif(NUMPY_FOUND)

mark_as_advanced(NUMPY_INCLUDE_DIR)
