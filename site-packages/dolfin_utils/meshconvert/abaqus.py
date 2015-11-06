# -*- coding: utf-8 -*-
"""Module for converting the  Abaqus mesh format."""

# Copyright (C) 2012 Arve Knudsen and Garth N/ Wells
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
# Modified by Simon Funke (surface export)

from __future__ import print_function
# TODO: The change to python 3 compatible iteration may have introduced performance
#       regressions here, with unnecessary list() applications in the below code.
from six import iterkeys, iteritems
import re
import csv
import numpy as np

from . import xml_writer

class State:
    Init, Unknown, Invalid, ReadHeading, ReadNodes, ReadCells, \
        ReadNodeSet, ReadCellSet, ReadSurfaceSet = list(range(9))

def convert(ifilename, handler):
    """ Convert from Abaqus.

    The Abaqus format first defines a node block, then there should be a number
    of elements containing these nodes.
    """

    # Dictionary of nodes (maps node id to coordinates)
    nodes = {}

    # Dictionary of elements (maps cell id to list of cell nodes)
    elems = {}

    # Lists of nodes for given name (key)
    node_sets = {}

    # Lists of cells for given name (key)
    cell_sets = {}

    # Lists of surfaces for given name (key) in the format:
    # {'SS1': [set(['SS1_S1', 'S1']), set(['SS1_S4', 'S4'])]},
    # where SS1 is the name of the surface, SS1_S1 is the name of the
    # cell list whose first face is to be selected, ...
    surface_sets = {}

    # Open file Abaqus file
    file = open(ifilename, 'r')
    csv_file = csv.reader(file, delimiter=',', skipinitialspace=True)

    node_set_name = None
    generate = None

    # Set intial state state
    state = State.Init

    # Read data from input file
    for l in csv_file:

        # Sanity check
        if (len(l) == 0): print("Ooops, zero length.")

        if l[0].startswith('**'): # Pass over comments
            continue
        elif l[0].startswith('*'): # Have a keyword
            state = State.Unknown

            if l[0].lower() == "*heading":
                state = State.ReadHeading

            elif l[0].lower() == "*part":
                part_name = _read_part_name(l)

            elif l[0].lower() == "*end part":
                state = State.Invalid

            elif l[0].lower() == "*node":
                node_set_name = _create_node_list_entry(node_sets, l)
                state = State.ReadNodes

            elif l[0].lower() == "*element":
                cell_type, cell_set_name = _read_element_keywords(cell_sets, l)
                state = State.ReadCells

            elif l[0].lower() == "*nset":
                node_set_name, generate = _read_nset_keywords(node_sets, l)
                state = State.ReadNodeSet

            elif l[0].lower() == "*elset":
                cell_set_name, generate = _read_elset_keywords(cell_sets, l)
                if generate:
                    print("WARNING: generation of *elsets not tested.")
                state = State.ReadCellSet

            elif l[0].lower() == "*surface":
                surface_set_name, generate = _read_surface_keywords(surface_sets, l)
                state = State.ReadSurfaceSet

            else:
                print("WARNING: unrecognised Abaqus input keyword:", l[0])
                state = State.Unknown

        else:

            if state == State.ReadHeading:
                model_name = _read_heading(l)

            elif state == State.ReadNodes:
                node_id = int(l[0]) - 1
                coords = [float(c) for c in l[1:]]
                nodes[node_id] = coords
                if node_set_name is not None:
                    node_sets[node_set_name].add(node_id)

            elif state == State.ReadCells:
                cell_id = int(l[0]) - 1
                cell_connectivity = [int(v) - 1 for v in l[1:]]
                elems[cell_id] = cell_connectivity
                if cell_set_name is not None:
                    cell_sets[cell_set_name].add(cell_id)

            elif state == State.ReadNodeSet:

                try:
                    if generate:
                        n0, n1, increment = l
                        node_range = list(range(int(n0) - 1, int(n1) - 1, int(increment)))
                        node_range.append(int(n1) - 1)
                        node_sets[node_set_name].update(node_range)
                    else:
                        # Strip empty term at end of list, if present
                        if l[-1] == '': l.pop(-1)
                        node_range = [int(n) - 1 for n in l]
                        node_sets[node_set_name].update(node_range)
                except:
                    print("WARNING: Non-integer node sets not yet supported.")

            elif state == State.ReadCellSet:
                try:
                    if generate:
                        n0, n1, increment = l
                        cell_range = list(range(int(n0) - 1, int(n1) - 1, int(increment)))
                        cell_range.append(int(n1) - 1)
                        cell_sets[cell_set_name].update(cell_range)
                    else:
                        # Strip empty term at end of list, if present
                        if l[-1] == '': l.pop(-1)
                        cell_range = [int(n) - 1 for n in l]
                        cell_sets[cell_set_name].update(cell_range)
                except:
                    print("WARNING: Non-integer element sets not yet supported.")

            elif state == State.ReadSurfaceSet:
                # Strip empty term at end of list, if present
                if l[-1] == '': l.pop(-1)
                surface_sets[surface_set_name].update([tuple(l)])

            elif state == State.Invalid: # part
                raise Exception("Inavlid Abaqus parser state..")


    # Close CSV object
    file.close()
    del csv_file

    # Write data to XML file
    # Note that vertices/cells must be consecutively numbered, which
    # isn't necessarily the case in Abaqus. Therefore we enumerate and
    # translate original IDs to sequence indexes if gaps are present.

    # FIXME
    handler.set_mesh_type("tetrahedron", 3)

    process_facets = len(surface_sets) > 0
    if process_facets:
        try:
            from dolfin import MeshEditor, Mesh
        except ImportError:
            _error("DOLFIN must be installed to handle Abaqus boundary regions")

        mesh = Mesh()
        mesh_editor = MeshEditor()
        mesh_editor.open(mesh, 3, 3)

    node_ids_order = {}
    # Check for gaps in vertex numbering
    node_ids = list(iterkeys(nodes))
    if len(node_ids) > 0:
        vertex_gap = (min(node_ids) != 0 or max(node_ids) != len(node_ids) - 1)
        for x, y in enumerate(node_ids):
            node_ids_order[y]= x  # Maps Abaqus IDs to Dolfin IDs
    else:
        vertex_gap = True

    # Check for gaps in cell numbering
    elemids = list(iterkeys(elems))
    if len(elemids) > 0:
        cell_gap = (min(elemids) != 0 or max(elemids) != len(elemids) - 1)
    else:
        cell_gap = True

    # Write vertices to XML file
    handler.start_vertices(len(nodes))
    if process_facets:
        mesh_editor.init_vertices_global(len(nodes), len(nodes))

    if not vertex_gap:

        for v_id, v_coords in list(iteritems(nodes)):
            handler.add_vertex(v_id, v_coords)
            if process_facets:
                mesh_editor.add_vertex(v_id, np.array(v_coords, dtype=np.float_))

    else:

        for idx, (v_id, v_coords) in enumerate(iteritems(nodes)):
            handler.add_vertex(idx, v_coords)
            if process_facets:
                mesh_editor.add_vertex(idx, np.array(v_coords, dtype=np.float_))

    handler.end_vertices()

    # Write cells to XML file
    handler.start_cells(len(elems))
    if process_facets:
        mesh_editor.init_cells_global(len(elems), len(elems))

    if not vertex_gap and not cell_gap:

        for c_index, c_data in list(iteritems(elems)):
            for v_id in c_data:
                if not (0 <= v_id < len(nodes)):
                    handler.error("Element %s references non-existent node %s" % (c_index, v_id))
            handler.add_cell(c_index, c_data)

            if process_facets:
                c_data_tmp = np.array(c_data)
                c_data_tmp.sort()
                mesh_editor.add_cell(c_index, np.array(c_data_tmp, dtype=np.uintp))


    elif not vertex_gap and cell_gap:

        for idx, (c_index, c_data) in enumerate(iteritems(elems)):
            for v_id in c_data:
                if not (0 <= v_id < len(nodes)):
                    handler.error("Element %s references non-existent node %s" % (c_index, v_id))
            handler.add_cell(idx, c_data)

            if process_facets:
                c_data_tmp = np.array(c_data)
                c_data_tmp.sort()
                mesh_editor.add_cell(idx, np.array(c_data_tmp, dtype=np.uintp))

    else:

        for idx, (c_id, c_data) in enumerate(iteritems(elems)):
            c_nodes = []
            for v_id in c_data:
                try: c_nodes.append(node_ids_order[v_id])
                except ValueError:
                    handler.error("Element %s references non-existent node %s" % (c_id, v_id))
            handler.add_cell(idx, c_nodes)

            if process_facets:
                c_nodes.sort()
                mesh_editor.add_cell(idx, np.array(c_nodes, dtype=np.uintp))

    handler.end_cells()

    # Write MeshValueCollections to XML file
    handler.start_domains()

    # Build a abaqus node ID -> dolfin cell ID map (which is not unique but that is irrelevant here)
    #                           and its local entity.
    if len(node_sets) > 0:
        node_cell_map = {}
        for c_dolfin_index, (c_index, c_data) in enumerate(iteritems(elems)):
            c_data_tmp = np.array(c_data)
            c_data_tmp.sort()
            for local_entity, n_index in enumerate(c_data_tmp):
                node_cell_map[n_index] = (c_dolfin_index, local_entity)

    # Write vertex/node sets
    dim = 0
    for value, (name, node_set) in enumerate(iteritems(node_sets)):
        handler.start_mesh_value_collection(name, dim, len(node_set), "uint")

        for node in node_set:
            try:
                cell, local_entity = node_cell_map[node]
                handler.add_entity_mesh_value_collection(dim, cell, value, local_entity=local_entity)
            except KeyError:
                print("Warning: Boundary references non-existent node %s" % node)
        handler.end_mesh_value_collection()

    # Write cell/element sets
    dim = 3
    for name, s in list(iteritems(cell_sets)):
        handler.start_mesh_value_collection(name, dim, len(s), "uint")
        for cell in s:
            handler.add_entity_mesh_value_collection(dim, cell, 0)
        handler.end_mesh_value_collection()

    # Write surface sets
    if process_facets:
        dim = 2
        nodes_facet_map = _nodes_facet_map(mesh)

        data = [int(0)] * mesh.num_facets()
        S1 = [0, 1, 2]
        S2 = [0, 3, 1]
        S3 = [1, 3, 2]
        S4 = [2, 3, 0]
        node_selector = {'S1': S1,
                         'S2': S2,
                         'S3': S3,
                         'S4': S4,
                         }

        for index, (name, s) in enumerate(iteritems(surface_sets)):
            cell_face_list = []
            for cell_set_name, face_index in s:
                cell_face_list += [(cell, face_index) for cell in cell_sets[cell_set_name]]

            for cell, face in cell_face_list:
                cell_nodes = elems[cell]
                # Extract the face nodes
                face_nodes = [cell_nodes[i] for i in node_selector[face]]
                dolfin_face_nodes = [node_ids_order[n] for n in face_nodes]
                dolfin_face_nodes.sort()
                # Convert the face_nodes to dolfin IDs
                face_id = nodes_facet_map[tuple(dolfin_face_nodes)]
                data[face_id] = index + 1

        # Create and initialise the mesh function
        handler.start_meshfunction("facet_region", dim, mesh.num_facets() )
        for index, physical_region in enumerate (data):
            handler.add_entity_meshfunction(index, physical_region)
        handler.end_meshfunction()


    handler.end_domains()

def _nodes_facet_map(mesh):
    # Now process the facet markers
    dim = 2
    mesh.init(dim, 0)
    facets_as_nodes = mesh.topology()(dim, 0)().reshape(mesh.num_facets(), 3)

    # Build the reverse map
    nodes_as_facets = {}
    for facet in range(mesh.num_facets()):
        nodes_as_facets[tuple(facets_as_nodes[facet,:])] = facet

    return nodes_as_facets


def _read_heading(l):
    return l[0].strip()


def _read_part_name(l):

    if (len(l) < 2): print("Ooops, length problem.")
    part_names = l[1].split('=')

    if (len(part_names) < 2): print("Ooops, part names length problem.")
    return part_names[1].strip()


def _create_node_list_entry(node_sets, l):

    # Check for node set name
    node_set_name = None
    if len(l) == 2:
        set_data = l[1].split('=')
        assert len(set_data) == 2, "wrong list length"
        if set_data[0].lower() == "nset":
            node_set_name = set_data[1]
            if node_set_name not in node_sets:
                node_sets[node_set_name] = set()
    return node_set_name

def _read_element_keywords(cell_sets, l):

    # Get element type and element set name
    element_type = None
    element_set_name = None
    for key in l[1:]:
        key_parts = key.split('=')
        key_name = key_parts[0].lower().strip()
        if key_name == "type":
            element_type = key_parts[1].lower().strip()
        elif key_name == "elset":
            element_set_name = key_parts[1].strip()

    # Test that element is supported
    check_element_support(element_type)

    # Add empty set to cell_sets dictionary
    if element_set_name:
        if element_set_name not in cell_sets:
            cell_sets[element_set_name] = set()

    return element_type, element_set_name


def _read_nset_keywords(node_sets, l):

    node_set_name = None
    generate = None

    # Get set name and add to dict
    set_data = l[1].split('=')
    assert len(set_data) == 2, "wrong list length, set name missing"
    assert set_data[0].lower() == "nset"
    node_set_name = set_data[1]
    if node_set_name not in node_sets:
        node_sets[node_set_name] = set()

    # Check for generate flag
    if len(l) == 3:
        if l[2].lower() == "generate":
           generate = True

    return node_set_name, generate


def _read_elset_keywords(sets, l):

    set_name = None
    generate = None

    # Get set name and add to dict
    set_data = l[1].split('=')
    assert len(set_data) == 2, "wrong list length, set name missing"
    assert set_data[0].lower() == "elset"
    set_name = set_data[1]
    if set_name not in sets: sets[set_name] = set()

    # Check for generate flag
    if len(l) == 3:
        if l[2].lower() == "generate":
            generate = True

    return set_name, generate

def _read_surface_keywords(sets, l):

    surface_name = None
    generate = None

    # Get surface name and add to dict
    surface_data = l[1].split('=')
    assert len(surface_data) == 2, "wrong list length, surface name missing"
    assert surface_data[0].lower() == "name"
    surface_name = surface_data[1]
    if surface_name not in sets: sets[surface_name] = set()

    generate = False
    return surface_name, generate


def check_element_support(element_type):
    supported_elements = ('c3d4',)
    if element_type.lower() not in supported_elements:
        raise Exception("Element type not supported.")
