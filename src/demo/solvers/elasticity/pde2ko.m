function h = pde2ko(p, e, t, u, v1, fname)
  
  u = v1;

  pdeformed = p + u;
  

  color1 = [0.0 1.0 0.0];
  color2 = [1.0 0.0 0.0];

  %Compute some useful information about the geometry

  %reconstruct edges

  %e = [0; 0];

  %j = 1;
  %for i = 1:size(t, 2)
  %  e(:, j + 0) = [t(1, i); t(2, i)];
  %  e(:, j + 1) = [t(2, i); t(3, i)];
  %  e(:, j + 2) = [t(1, i); t(3, i)];
  %  j = j + 3;
  %end

  %emin = 0;

  %compute minimum edge length

  %for i = 1:size(e, 2)
  %  elen = norm(p(:, e(1, i)) - p(:, e(2, i)));
  %  if(i == 1)
  %    emin = elen;
  %  else
  %    emin = min(elen, emin);
  %  end
  %end

  %emin
  %r = emin / 4;
  r = 0.08;

  pmax = max(p');
  pmin = min(p');

  %umin = min(u);
  %umax = max(u);

  %umax
  %umin

  %urange = umax - umin;

  %if(urange == 0)
  %  urange = 1;
  %end


  %offset = urange / 100;
  %offset

  %center = (pmax + pmin) / 2
  %centeru = (umax + umin) / 2

  %Assume that u is xydata
  
  fd = fopen(fname, 'w');

  fprintf(fd, '<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n');
  fprintf(fd,
   '<physical xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:noNamespaceSchemaLocation=\"physical-schema.xml\">\n');

  fprintf(fd, '<material>\n');
  fprintf(fd, '<name>nylon</name>\n');
  fprintf(fd, '<young>1.000000e+05</young>\n');
  fprintf(fd, '<density>1.000000e+03</density>\n');
  fprintf(fd, '<damp>1.000e+01</damp>\n');
  fprintf(fd, '</material>\n');


  size(u);

  for i = 1:size(p, 2)
    

    %pdeformed = p(:, i) + u(:, i)
    %pdeformed = p(:, i);

    fprintf(fd, '<element>\n');
    fprintf(fd, '<name>%d</name>\n', i);
    fprintf(fd, '<materialName>nylon</materialName>\n');
    fprintf(fd, '<radius>%f</radius>\n', r);
    fprintf(fd, '<position>\n', r);
    %fprintf(fd, '<x>%f</x>\n', p(1, i));
    %fprintf(fd, '<y>%f</y>\n', p(2, i));
    %fprintf(fd, '<z>%f</z>\n', 0);
    fprintf(fd, '<x>%f</x>\n', pdeformed(1, i));
    fprintf(fd, '<y>%f</y>\n', pdeformed(2, i));
    fprintf(fd, '<z>%f</z>\n', pdeformed(3, i));
    fprintf(fd, '</position>\n');
    fprintf(fd, '<velocity>\n');
    fprintf(fd, '<x>%f</x>\n', 0);
    fprintf(fd, '<y>%f</y>\n', 0);
    fprintf(fd, '<z>%f</z>\n', 0);
    fprintf(fd, '</velocity>\n');
    fprintf(fd, '<artificial>%s</artificial>\n', 'false');
    fprintf(fd, '</element>\n');
  end

  for i = 1:size(t, 2)
    
    fprintf(fd, '<connection>\n');
    fprintf(fd, '<element1>%d</element1>\n', t(1, i));
    fprintf(fd, '<element2>%d</element2>\n', t(2, i));
    fprintf(fd, '<length>%f</length>\n', norm(pdeformed(:, t(1, i)) - pdeformed(:, t(2, i))));
    fprintf(fd, '<transient>false</transient>\n');
    fprintf(fd, '</connection>\n');

    fprintf(fd, '<connection>\n');
    fprintf(fd, '<element1>%d</element1>\n', t(1, i));
    fprintf(fd, '<element2>%d</element2>\n', t(3, i));
    fprintf(fd, '<length>%f</length>\n', norm(pdeformed(:, t(1, i)) - pdeformed(:, t(3, i))));
    fprintf(fd, '<transient>false</transient>\n');
    fprintf(fd, '</connection>\n');

    fprintf(fd, '<connection>\n');
    fprintf(fd, '<element1>%d</element1>\n', t(1, i));
    fprintf(fd, '<element2>%d</element2>\n', t(4, i));
    fprintf(fd, '<length>%f</length>\n', norm(pdeformed(:, t(1, i)) - pdeformed(:, t(4, i))));
    fprintf(fd, '<transient>false</transient>\n');
    fprintf(fd, '</connection>\n');

    fprintf(fd, '<connection>\n');
    fprintf(fd, '<element1>%d</element1>\n', t(2, i));
    fprintf(fd, '<element2>%d</element2>\n', t(3, i));
    fprintf(fd, '<length>%f</length>\n', norm(pdeformed(:, t(2, i)) - pdeformed(:, t(3, i))));
    fprintf(fd, '<transient>false</transient>\n');
    fprintf(fd, '</connection>\n');

    fprintf(fd, '<connection>\n');
    fprintf(fd, '<element1>%d</element1>\n', t(2, i));
    fprintf(fd, '<element2>%d</element2>\n', t(4, i));
    fprintf(fd, '<length>%f</length>\n', norm(pdeformed(:, t(2, i)) - pdeformed(:, t(4, i))));
    fprintf(fd, '<transient>false</transient>\n');
    fprintf(fd, '</connection>\n');

    fprintf(fd, '<connection>\n');
    fprintf(fd, '<element1>%d</element1>\n', t(3, i));
    fprintf(fd, '<element2>%d</element2>\n', t(4, i));
    fprintf(fd, '<length>%f</length>\n', norm(pdeformed(:, t(3, i)) - pdeformed(:, t(4, i))));
    fprintf(fd, '<transient>false</transient>\n');
    fprintf(fd, '</connection>\n');
  end


  fprintf(fd, '</physical>\n');

  fclose(fd);
