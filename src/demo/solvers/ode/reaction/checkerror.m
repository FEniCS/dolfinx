% Copyright (C) 2005 Anders Logg.
% Licensed under the GNU GPL Version 2.
%
% Load solution at end time and compute the error in
% the maximum norm by comparing to the reference solution.

load solution.data
load reference.data

e = max(abs(solution - reference));

printf('Error %.3e', e)
