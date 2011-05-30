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

execute_process(COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/generate_main_index.py" "${CMAKE_CURRENT_BINARY_DIR}/${DOLFIN_CPP_SPHINX_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/${DOLFIN_PYTHON_SPHINX_SOURCE_DIR}" "${DOLFIN_VERSION}")

# Add target api_doc: generate reST files from code for cpp and
# python. This takes a while.
add_custom_target(api_doc
  COMMAND ${PYTHON_EXECUTABLE} generate_api_doc.py
    "${DOLFIN_SOURCE_DIR}" api "${DOLFIN_VERSION}"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    api/cpp/programmers-reference
    "${DOLFIN_CPP_SPHINX_SOURCE_DIR}/programmers-reference"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    api/python/programmers-reference
    "${DOLFIN_PYTHON_SPHINX_SOURCE_DIR}/programmers-reference"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# Add target demo_doc: copy demo code and doc files into sphinx
# projects and add additional index files.
add_custom_target(demo_doc
  COMMAND ${PYTHON_EXECUTABLE} copy_and_split_demo_doc.py
    "${DOLFIN_SOURCE_DIR}/demo"
    "${DOLFIN_CPP_SPHINX_SOURCE_DIR}/demo"
    "${DOLFIN_PYTHON_SPHINX_SOURCE_DIR}/demo"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# Add target [cpp|python]_html: run sphinx in [cpp|python]
# sphinx projects
if (SPHINX_FOUND)
  message(STATUS "Enabling generation of html-documentation")
  add_custom_target(cpp_html
    COMMAND ${MAKE_PROGRAM} html
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${DOLFIN_CPP_SPHINX_DIR}")

  add_custom_target(python_html
    COMMAND ${MAKE_PROGRAM} html
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${DOLFIN_PYTHON_SPHINX_DIR}")

  # Add target html: run sphinx in both cpp and python
  add_custom_target(html)
  add_dependencies(html cpp_html python_html)

  # Add target doc: Generate all doc (api and demos) and run sphinx
  # to generate html
  add_custom_target(all_doc)
  add_dependencies(all_doc demo_doc api_doc)

  # FIXME: Need to run all_doc before html (order of targets not
  # respected)
  add_custom_target(doc)
  add_dependencies(doc all_doc html)

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
