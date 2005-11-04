% Copyright (C) 2005 Anders Logg.
% Licensed under the GNU GPL Version 2.
%
% Load solution at end time and compute the error in
% the maximum norm by comparing to the reference solution.

load solution.data
load reference-fixedpoint.data

e = max(abs(solution - reference_fixedpoint));

printf('Error: %.3e\n', e)
