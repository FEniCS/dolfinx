# -*- coding: utf-8 -*-
# Copyright (C) 2018 Chris N Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import subprocess
import os

def pkgconfig_query(s):
    pkg_config_exe = os.environ.get('PKG_CONFIG', None) or 'pkg-config'
    cmd = [pkg_config_exe] + s.split()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = proc.communicate()
    rc = proc.returncode
    return (rc, out.rstrip().decode('utf-8'))

def exists(pkg_name):
    return (pkgconfig_query("--exists " + pkg_name)[0] == 0)

def parse(package):
    parse_map = {'D': 'define_macros',
                 'I': 'include_dirs',
                 'L': 'library_dirs',
                 'l': 'libraries'}

    result = {x:[] for x in parse_map.values()}

    # Execute the query to pkg-config and clean the result.
    out = pkgconfig_query(package + ' --cflags --libs')[1]
    out = out.replace('\\"', '')

    # Iterate through each token in the output.
    for token in out.split():
        key = parse_map[token[1]]
        t = token[2:].strip()
        if (key == 'define_macros'):
            t = tuple(t.split('='))
            if len(t) == 1:
                t += '',
        result[key].append(t)

    return result
