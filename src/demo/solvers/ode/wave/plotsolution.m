% Copyright (C) 2005 Anders Logg.
% Licensed under the GNU GPL Version 2.

% Load solution
solutionu

% Plot solution
figure(1)
pdesurf(points, cells, u{1})

figure(2)
pdesurf(points, cells, u{45})
