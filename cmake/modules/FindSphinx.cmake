# - Try to find Sphinx (sphinx-build)

# Once done this will define
#
#   SPHINX_FOUND      - system has Spinx
#   SPHINX_EXECUTABLE - sphinx-build
#   SPHINX_VERSION    - Major.Minor.Micro version of Sphinx

message(STATUS "Checking for package 'Sphinx'")

find_program(SPHINX_EXECUTABLE sphinx-build
  HINTS
  $ENV{SPHINX_DIR}
  PATH_SUFFIXES bin
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Sphinx
  "The program sphinx-build could not be found."
  SPHINX_EXECUTABLE
)

if (SPHINX_FOUND)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import sphinx; print sphinx.__version__"
    OUTPUT_VARIABLE SPHINX_VERSION
    RESULT_VARIABLE SPHINX_VERSION_NOT_FOUND)
endif()

if (SPHINX_VERSION)
  set (SPHINX_VERSION ${SPHINX_VERSION} CACHE STRING "Sphinx version")
endif()

mark_as_advanced(
  SPHINX_EXECUTABLE
)
