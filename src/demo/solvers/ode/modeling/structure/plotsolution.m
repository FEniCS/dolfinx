% Copyright (C) 2004 Johan Hoffman and Anders Logg.
% Licensed under the GNU GPL Version 2.

primal

n = size(u,1)/4;
p = round(1/2 + sqrt(n/2 - 1/4));
M = size(u,2);

figure(1)

for m = 1:M

  clf
  
  for i=1:n
    
    x = u(2*(i-1) + 1, m);
    y = u(2*(i-1) + 2, m);

    if ( i <= p*p )
      plot(x, y, 'o')
    else
      plot(x, y, 'x')
    end
      
    hold on
    
  end

  axis([-0.1 1.1 -0.1 1.1])
  axis equal
  
  disp(['t = ' num2str(t(m)) ' (Press any key to continue)'])
  pause
  
end

figure(2)

% Lower left particle
x1 = u(1:2,:);

% Upper right particle
x2 = u(2*p^2-1:2*p^2,:);

% Radius of structure
r = sqrt(sum((x1 - x2).^2,1));

plot(t, r)
