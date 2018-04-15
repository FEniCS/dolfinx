# Copyright (C) 2010-16 Garth N. Wells
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Generate CMakeLists.txt files in demo and test directories
# This script should be run from the top level directory.
#
# Modified by Anders Logg 2013, 2014

import os
import subprocess

cmakelists_str = \
"""# This file is automatically generated by running
#
#     cmake/scripts/generate-cmakefiles
#
# Require CMake 3.5
cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME %(project_name)s)
project(${PROJECT_NAME})

# Set CMake behavior
cmake_policy(SET CMP0004 NEW)

# Get DOLFIN configuration data (DOLFINConfig.cmake must be in
# DOLFIN_CMAKE_CONFIG_PATH)
find_package(DOLFIN REQUIRED)

include(${DOLFIN_USE_FILE})

# Default build type (can be overridden by user)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
      "Choose the type of build, options are: Debug MinSizeRel Release RelWithDebInfo." FORCE)
endif()

# Do not throw error for 'multi-line comments' (these are typical in
# rst which includes LaTeX)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wno-comment" HAVE_NO_MULTLINE)
if (HAVE_NO_MULTLINE)
  set(CMAKE_CXX_FLAGS "-Wno-comment ${CMAKE_CXX_FLAGS}")
endif()

# Executable
%(executables)s

# Target libraries
%(target_libraries)s

# Test targets
set(test_parameters -np 3 "./${PROJECT_NAME}")
add_test(NAME ${PROJECT_NAME}_mpi COMMAND "mpirun" ${test_parameters})
add_test(NAME ${PROJECT_NAME}_serial COMMAND ${PROJECT_NAME})
"""

executable_str = "add_executable(%s %s)"
target_link_libraries_str = "target_link_libraries(%s dolfin)"

# Subdirectories
sub_directories = ['demo', 'bench']
# Prefix map for subdirectories
executable_prefixes = dict(demo="demo_", bench="bench_")

# Main file name map for subdirectories
main_file_names = dict(
    demo=set(["main.cpp", "main.cpp.rst"]), bench=set(["main.cpp"]))

# Projects that use custom CMakeLists.txt (shouldn't overwrite)
#exclude_projects = [os.path.join('demo', 'undocumented', 'plot-qt')]
exclude_projects = []


def generate_cmake_files(subdirectory, generated_files):
    """Search for C++ code and write CMakeLists.txt files"""
    cwd = os.getcwd()
    executable_prefix = executable_prefixes[subdirectory]
    main_file_name = main_file_names[subdirectory]
    for root, dirs, files in os.walk(cwd + "/" + subdirectory):

        cpp_files = set()
        c_files = set()
        executable_names = set()

        program_dir = root
        program_name = os.path.split(root)[-1]

        skip = False
        for exclude in exclude_projects:
            if exclude in root:
                skip = True

        if skip:
            print("Skipping custom CMakeLists.txt file:", root)
            continue

        name_forms = dict(
            project_name=executable_prefix + program_name,
            executables="NOT_SET",
            target_libraries="NOT_SET")
        for f in os.listdir(program_dir):
            filename, extension = os.path.splitext(f)
            if extension == ".cpp":
                cpp_files.add(f)
            if extension == ".c":
                c_files.add(f)
            if ".cpp.rst" in f:
                cpp_files.add(filename)

        # If no .cpp, continue
        if not cpp_files:
            continue

        # Name of demo and cpp source files
        if not main_file_name.isdisjoint(cpp_files):

            # If directory contains a main file we assume that only one
            # executable should be generated for this directory and all
            # other .cpp files should be linked to this
            name_forms["executables"] = executable_str % \
                                        ("${PROJECT_NAME}",
                                         ' '.join(cpp_files | c_files))
            name_forms["target_libraries"] = target_link_libraries_str % \
                                             "${PROJECT_NAME}"
        else:
            # If no main file in source files, we assume each source
            # should be compiled as an executable
            name_forms["executables"] = "\n".join(
                executable_str % (executable_prefix + f.replace(".cpp", ""), f)
                for f in cpp_files)
            name_forms["target_libraries"] = "\n".join(
                target_link_libraries_str % (
                    executable_prefix + f.replace(".cpp", ""))
                for f in cpp_files)

        # Check for duplicate executable names
        if program_name not in executable_names:
            executable_names.add(program_name)
        else:
            print(
                "Warning: duplicate executable names found when generating CMakeLists.txt files."
            )

        # Write file
        filename = os.path.join(program_dir, "CMakeLists.txt")
        generated_files.append(filename)
        with open(filename, "w") as f:
            f.write(cmakelists_str % name_forms)


# Generate CMakeLists.txt files for all subdirectories
generated_files = []
for subdirectory in sub_directories:
    generate_cmake_files(subdirectory, generated_files)

# Print list of generated files
print("The following files were generated:")
print("\n".join(generated_files))
