project(dolfin-doc)

message(STATUS "Enabling generation of documentation.")

# Copy most of doc directory to build directory
file(COPY . DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  FILES_MATCHING
  PATTERN "*.rst"
  PATTERN "*.py"
  PATTERN "*.conf"
  PATTERN "*.html"
  PATTERN "*.png"
  PATTERN "*.css*"
  PATTERN "Makefile"
  PATTERN "CMakeFiles" EXCLUDE)

# Set directories for where sphinx projects and their source
# directories are located
set(DOLFIN_CPP_SPHINX_DIR "sphinx-cpp")
set(DOLFIN_CPP_SPHINX_SOURCE_DIR "${DOLFIN_CPP_SPHINX_DIR}/source")
set(DOLFIN_PYTHON_SPHINX_DIR "sphinx-python")
set(DOLFIN_PYTHON_SPHINX_SOURCE_DIR "${DOLFIN_PYTHON_SPHINX_DIR}/source")

# Initialize separate cpp and python projects
message(STATUS "Initializing sphinx projects")

file(COPY "${CMAKE_CURRENT_BINARY_DIR}/sphinx-common/"
  DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/${DOLFIN_CPP_SPHINX_DIR}")
file(COPY "${CMAKE_CURRENT_BINARY_DIR}/sphinx-common/"
  DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/${DOLFIN_PYTHON_SPHINX_DIR}")

configure_file("${CMAKE_CURRENT_BINARY_DIR}/quick_reference_python.rst"
  "${CMAKE_CURRENT_BINARY_DIR}/${DOLFIN_PYTHON_SPHINX_SOURCE_DIR}/quick_reference.rst" COPYONLY)
configure_file("${CMAKE_CURRENT_BINARY_DIR}/quick_reference_cpp.rst"
  "${CMAKE_CURRENT_BINARY_DIR}/${DOLFIN_CPP_SPHINX_SOURCE_DIR}/quick_reference.rst" COPYONLY)

execute_process(COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/generate_main_index.py" "${CMAKE_CURRENT_BINARY_DIR}/${DOLFIN_CPP_SPHINX_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/${DOLFIN_PYTHON_SPHINX_SOURCE_DIR}" "${DOLFIN_VERSION}")

# Add target doc_api: generate reST files from code for cpp and
# python. This takes a while.
add_custom_target(doc_api
  COMMAND ${PYTHON_EXECUTABLE} generate_api_doc.py
    "${DOLFIN_SOURCE_DIR}" api "${DOLFIN_VERSION}"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    api/cpp/programmers-reference
    "${DOLFIN_CPP_SPHINX_SOURCE_DIR}/programmers-reference"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    api/python/programmers-reference
    "${DOLFIN_PYTHON_SPHINX_SOURCE_DIR}/programmers-reference"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# Add target doc_demo: copy demo code and doc files into sphinx
# projects and add additional index files.
add_custom_target(doc_demo
  COMMAND ${PYTHON_EXECUTABLE} copy_and_split_demo_doc.py
    "${DOLFIN_SOURCE_DIR}/demo"
    "${DOLFIN_CPP_SPHINX_SOURCE_DIR}/demo"
    "${DOLFIN_PYTHON_SPHINX_SOURCE_DIR}/demo"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# Add target doc_html_[cpp|python]: run sphinx in [cpp|python] sphinx
# projects
if (SPHINX_FOUND)
  message(STATUS "Enabling generation of html-documentation")
  add_custom_target(doc_html_cpp
    COMMAND ${MAKE_PROGRAM} html
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${DOLFIN_CPP_SPHINX_DIR}")

  add_custom_target(doc_html_python
    COMMAND ${MAKE_PROGRAM} html
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${DOLFIN_PYTHON_SPHINX_DIR}")

  # Add target doc_html: run sphinx in both cpp and python
  add_custom_target(doc_html)
  add_dependencies(doc_html doc_html_cpp doc_html_python)

  # Add target doc_generate_all: Generate all doc (api and demos)
  add_custom_target(doc_generate_all)
  add_dependencies(doc_generate_all doc_demo doc_api)

  # FIXME: Want to run doc_generate_all before doc_html, but the order
  # of the targets is necessarily respected here.
  add_custom_target(doc)
  add_dependencies(doc doc_generate_all doc_html)

  # Add target: doc_install Copy generated sphinx projects to
  # DOLFIN_DOC_DIR
  add_custom_target(doc_install
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_CURRENT_BINARY_DIR}/${DOLFIN_CPP_SPHINX_DIR}"
      "${CMAKE_INSTALL_PREFIX}/${DOLFIN_DOC_DIR}/${DOLFIN_CPP_SPHINX_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_CURRENT_BINARY_DIR}/${DOLFIN_PYTHON_SPHINX_DIR}"
      "${CMAKE_INSTALL_PREFIX}/${DOLFIN_DOC_DIR}/${DOLFIN_PYTHON_SPHINX_DIR}")
  add_dependencies(doc_install doc)

endif()
