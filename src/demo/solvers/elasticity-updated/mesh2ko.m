function h = mesh2ko(p, e, t, fname)
  
  color1 = [0.0 1.0 0.0];
  color2 = [1.0 0.0 0.0];

  r = 0.2;

  pmax = max(p');
  pmin = min(p');

  center = (pmax + pmin) / 2

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


  for i = 1:size(p, 2)
    
    
    fprintf(fd, '<element>\n');
    fprintf(fd, '<name>%d</name>\n', i);
    fprintf(fd, '<materialName>nylon</materialName>\n');
    fprintf(fd, '<radius>%f</radius>\n', r);
    fprintf(fd, '<position>\n', r);
    %fprintf(fd, '<x>%f</x>\n', p(1, i));
    %fprintf(fd, '<y>%f</y>\n', p(2, i));
    %fprintf(fd, '<z>%f</z>\n', 0);
    fprintf(fd, '<x>%f</x>\n', p(1, i));
    fprintf(fd, '<y>%f</y>\n', p(2, i));
    fprintf(fd, '<z>%f</z>\n', p(3, i));
    fprintf(fd, '</position>\n');
    fprintf(fd, '<velocity>\n');
    fprintf(fd, '<x>%f</x>\n', 0);
    fprintf(fd, '<y>%f</y>\n', 0);
    fprintf(fd, '<z>%f</z>\n', 0);
    fprintf(fd, '</velocity>\n');
    fprintf(fd, '<artificial>%s</artificial>\n', 'false');
    fprintf(fd, '</element>\n');
  end

  fprintf(fd, '</physical>\n');

  fclose(fd);
