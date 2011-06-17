# - Try to find Sphinx (sphinx-build)

# Once done this will define
#
#   SPHINX_FOUND      - system has Sphinx
#   SPHINX_EXECUTABLE - full path to the Sphinx documentation generator tool
#   SPHINX_VERSION    - the version of Sphinx which was found, e.g. "1.0.7"

message(STATUS "Checking for package 'Sphinx'")

# Make sure Python is available
if (NOT PYTHON_EXECUTABLE)
  find_package(PythonInterp)
endif()

# Try to find sphinx-build
find_program(SPHINX_EXECUTABLE sphinx-build
  HINTS ${SPHINX_DIR} $ENV{SPHINX_DIR}
  PATH_SUFFIXES bin
  DOC "Sphinx documentation generator tool"
)

if (SPHINX_EXECUTABLE)
  # Try to check Sphinx version by importing Sphinx
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import sphinx; print sphinx.__version__"
    OUTPUT_VARIABLE SPHINX_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if (Sphinx_FIND_VERSION)
    # Check if version found is >= required version
    if (NOT "${SPHINX_VERSION}" VERSION_LESS "${Sphinx_FIND_VERSION}")
      set(SPHINX_VERSION_OK TRUE)
    endif()
  else()
    # No specific version of Sphinx is requested
    set(SPHINX_VERSION_OK TRUE)
  endif()
endif()

mark_as_advanced(
  SPHINX_EXECUTABLE
  SPHINX_VERSION
  SPHINX_VERSION_OK
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Sphinx DEFAULT_MSG
  SPHINX_EXECUTABLE SPHINX_VERSION_OK)
