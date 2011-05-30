# - Try to find Sphinx (sphinx-build)

# Once done this will define
#
#   SPHINX_FOUND      - system has Spinx
#   SPHINX_EXECUTABLE - sphinx-build
#   SPHINX_VERSION    - Major.Minor.Micro version of Sphinx

message(STATUS "Checking for package 'Sphinx'")

# Try to find sphinx-build
find_program(SPHINX_EXECUTABLE sphinx-build
  HINTS
  $ENV{SPHINX_DIR}
  PATH_SUFFIXES bin
)

# Try to check Sphinx version by importing Sphinx
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import sphinx; print sphinx.__version__"
  OUTPUT_VARIABLE SPHINX_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE SPHINX_VERSION_NOT_FOUND)

mark_as_advanced(
  SPHINX_EXECUTABLE
  SPHINX_VERSION
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Sphinx
  "The program sphinx-build could not be found."
  SPHINX_EXECUTABLE
  SPHINX_VERSION
  )

