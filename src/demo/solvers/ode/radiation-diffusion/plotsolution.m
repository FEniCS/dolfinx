% Load data
solution
timesteps

% Plot mesh
figure(1)
clf
pdemesh(points{1}, edges{1}, cells{1})

% Plot solution
for n=1:size(u1, 2)
  
  figure(2)
  clf
  subplot(2, 1, 1)
  pdesurf(points{1}, cells{1}, u1{n})
  shading faceted
  subplot(2, 1, 2)
  pdesurf(points{1}, cells{1}, u2{n})
  shading faceted
  disp('Press any key to continue')
  pause
  
end
