% Copyright (C) 2005 Johan Hoffman and Anders Logg.
% Licensed under the GNU GPL Version 2.

% Load data
solutionu
solutionv
timesteps

% Get minimum and maximum for u and v
m = length(u);
vmin = 0; vmax = 0;
umin = 0; umax = 0;
for i = 1:m
  umin = min(umin, min(u{i}));
  umax = max(umax, max(u{i}));
  vmin = min(vmin, min(v{i}));
  vmax = max(vmax, max(v{i}));
end

% Plot solution
clf
for i = 1:m
  
  subplot(2, 2, 1)
  pdesurf(points, cells, u{i})
  axis([0 1 0 1 umin umax])
  
  subplot(2, 2, 2)
  pdesurf(points, cells, v{i})
  axis([0 1 0 1 vmin vmax])
  
  subplot(2, 2, 3)
  pdesurf(points, cells, u{i})
  view(2)
  
  subplot(2, 2, 4)
  pdesurf(points, cells, v{i})
  view(2)
  
  disp('Press any key to continue')
  pause
    
end
