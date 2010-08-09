set(DOLFIN_PYTHON_FOUND 0)

if(APPLE)
  find_program(PYTHON_EXECUTABLE
    NAMES python
    )

  if(PYTHON_EXECUTABLE)
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c "import sys; sys.stdout.write(sys.prefix + '/include/python' + sys.version[:3])"
      OUTPUT_VARIABLE PYTHON_INCLUDE_PATH
      )
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c "import sys; sys.stdout.write(sys.exec_prefix + '/Python')"
      OUTPUT_VARIABLE PYTHON_LIBRARY
      )

    set(DOLFIN_PYTHON_INCLUDE_PATH
      ${PYTHON_INCLUDE_PATH} CACHE PATH "Path to the directory containg Python header files."
      )

    set(DOLFIN_PYTHON_LIBRARY
      ${PYTHON_LIBRARY} CACHE FILEPATH "Path to the Python link library."
      )

    mark_as_advanced(DOLFIN_PYTHON_INCLUDE_PATH)
    mark_as_advanced(DOLFIN_PYTHON_LIBRARY)
    mark_as_advanced(PYTHON_EXECUTABLE)

    if(DOLFIN_PYTHON_INCLUDE_PATH AND DOLFIN_PYTHON_LIBRARY)
      set(DOLFIN_PYTHON_INCLUDE_DIRS
	${DOLFIN_PYTHON_INCLUDE_PATH}
	)

      set(DOLFIN_PYTHON_LIBS
	${DOLFIN_PYTHON_LIBRARY}
	)

      set(DOLFIN_PYTHON_FOUND 1)
    endif(DOLFIN_PYTHON_INCLUDE_PATH AND DOLFIN_PYTHON_LIBRARY)
  endif(PYTHON_EXECUTABLE)
else(APPLE)
  include(FindPythonInterp)
  include(FindPythonLibs)

  if(PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND)
    set(DOLFIN_PYTHON_INCLUDE_DIRS
      ${PYTHON_INCLUDE_PATH}
      )

    set(DOLFIN_PYTHON_LIBS
      ${PYTHON_LIBRARY}
      )

    mark_as_advanced(PYTHON_INCLUDE_PATH)
    mark_as_advanced(PYTHON_LIBRARY)
    mark_as_advanced(PYTHON_DEBUG_LIBRARIES)
    mark_as_advanced(PY_MODULES_LIST)
    mark_as_advanced(PY_STATIC_MODULES_LIST)

    set(DOLFIN_PYTHON_FOUND 1)
  endif(PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND)
endif(APPLE)
