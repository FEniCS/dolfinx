function h = pdeplot(p, e, t, p1, u)

% Plots the scalar solution u on the mesh described by p, e, t (points, edges,
% triangles).
%
% Copyright (C) 2004 Johan Jansson.
% Licensed under the GNU GPL Version 2.

  
  %u = v1;

  color1 = [0.0 1.0 0.0];
  color2 = [1.0 0.0 0.0];

  %Compute some useful information about the geometry

  pmax = max(p');
  pmin = min(p');

  umin = min(u);
  umax = max(u);

  urange = umax - umin;

  if(urange == 0)
    urange = 1;
  end


  %offset = max([umax / 100 1e-3]);
  offset = urange / 100;
%  offset

%  center = [(max(p(1, :)) + min(p(1, :))) / 2
%	    (max(p(2, :)) + min(p(2, :))) / 2]; 
%  centeru = (max(u) + min(u)) / 2;

  center = (pmax + pmin) / 2;
  centeru = (umax + umin) / 2;

  %Assume that u is xydata
  
  fname = tempname;
  %[fd, fdout, pid] = popen2('ivview', 'w');
  [fd, fdout, pid] = popen2('ivview', '');
  %[fd, fdout, pid] = popen2('SceneViewer -', '');
  %fd = fopen(fname, 'w');
  
  fprintf(fd, '#Inventor V2.1 ascii\n\n');

  fprintf(fd, 'Separator\n');
  fprintf(fd, '{\n');

  fprintf(fd, ' OrthographicCamera\n');
  fprintf(fd, ' {\n');
  fprintf(fd, '  position %f %f %f\n', 0, 0, 5);
  fprintf(fd, '  focalDistance %f\n', 5);
  fprintf(fd, ' }\n');


  fprintf(fd, ' ShapeHints\n');
  fprintf(fd, ' {\n');
  fprintf(fd, '  creaseAngle 1.5\n');
  fprintf(fd, '  vertexOrdering COUNTERCLOCKWISE\n');
  fprintf(fd, ' }\n');

  fprintf(fd, ' Scale\n');
  fprintf(fd, ' {\n');
  fprintf(fd, '  scaleFactor %f %f %f\n', 1.5 / (2 * center(1)), 1.5 / (2 * center(2)), 1 / (urange));

  fprintf(fd, ' }\n');

  fprintf(fd, ' Translation\n');
  fprintf(fd, ' {\n');
  fprintf(fd, '  translation %f %f %f\n', -center(1), -center(2), -centeru);
  fprintf(fd, ' }\n');

  fprintf(fd, ' LightModel\n');
  fprintf(fd, ' {\n');
%  fprintf(fd, ' model PHONG\n');
  fprintf(fd, '  model BASE_COLOR\n');
  fprintf(fd, ' }\n');


  fprintf(fd, 'Separator\n');
  fprintf(fd, '{\n');
  fprintf(fd, ' Font\n');
  fprintf(fd, ' {\n');
  fprintf(fd, '  size 20\n');
  fprintf(fd, ' }\n');
  fprintf(fd, ' Translation\n');
  fprintf(fd, ' {\n');
  fprintf(fd, '  translation 0.0 1.05 0\n');
  fprintf(fd, ' }\n');
  fprintf(fd, ' Text2\n');
  fprintf(fd, ' {\n');
  fprintf(fd, '  string \"Title\"\n');
  fprintf(fd, ' }\n');
  fprintf(fd, '}\n');


  fprintf(fd, ' Material\n');
  fprintf(fd, ' {\n');
  fprintf(fd, '  ambientColor 0.5 0.5 0.5\n');
  fprintf(fd, '  specularColor 0.8 0.8 0.8\n');
  fprintf(fd, '  diffuseColor\n');
  fprintf(fd, '  [\n');

  for i = 1:size(u, 1)
    value = (u(i) - umin) / urange;
    color = (1 - value) * color1 + value * color2;
    %color

    if(i < size(u, 1))
      fprintf(fd, '   %f %f %f,\n', color(1), color(2), color(3));
    else
      fprintf(fd, '   %f %f %f\n', color(1), color(2), color(3));
    end
  end

  fprintf(fd, '  ]\n');
  fprintf(fd, ' }\n');

  fprintf(fd, ' MaterialBinding\n');
  fprintf(fd, ' {\n');
  fprintf(fd, '  value PER_VERTEX_INDEXED\n');
  fprintf(fd, ' }\n');


  fprintf(fd, ' Coordinate3\n');
  fprintf(fd, ' {\n');
  fprintf(fd, '  point\n');
  fprintf(fd, '  [\n');
	  
  for i = 1:size(p, 2)
    if(i < size(p, 2))
      fprintf(fd, '   %f %f %f,\n', p(1, i), p(2, i), u(i));
    else
      fprintf(fd, '   %f %f %f\n', p(1, i), p(2, i), u(i));
    end
  end

  fprintf(fd, '  ]\n');
  fprintf(fd, ' }\n');

  fprintf(fd, ' IndexedFaceSet\n');
  fprintf(fd, ' {\n');

  fprintf(fd, '  coordIndex\n');
  fprintf(fd, '  [\n');

  for i = 1:size(t, 2)
    if(i < size(t, 2))
      fprintf(fd, '   %d, %d, %d, -1,\n', t(1, i) - 1, t(2, i) - 1, t(3, i) - 1);
    else
      fprintf(fd, '   %d, %d, %d, -1\n', t(1, i) - 1, t(2, i) - 1, t(3, i) - 1);
    end
  end

  fprintf(fd, '  ]\n');
  fprintf(fd, ' }\n');


  fprintf(fd, ' Material\n');
  fprintf(fd, ' {\n');
  fprintf(fd, '  diffuseColor    0.1 0.1 0.1\n');
  fprintf(fd, ' }\n');


  fprintf(fd, ' MaterialBinding\n');
  fprintf(fd, ' {\n');
  fprintf(fd, '  value OVERALL\n');
  fprintf(fd, ' }\n');

  fprintf(fd, ' Coordinate3\n');
  fprintf(fd, ' {\n');
  fprintf(fd, '  point\n');
  fprintf(fd, '  [\n');
	  
  for i = 1:size(p, 2)
    if(i < size(p, 2))
      fprintf(fd, '   %f %f %f,\n', p(1, i), p(2, i), u(i) + offset);
    else
      fprintf(fd, '   %f %f %f\n', p(1, i), p(2, i), u(i) + offset);
    end
  end

  fprintf(fd, '  ]\n');
  fprintf(fd, ' }\n');



  fprintf(fd, ' IndexedLineSet\n');
  fprintf(fd, ' {\n');
  fprintf(fd, '  coordIndex\n');
  fprintf(fd, '  [\n');

  for i = 1:size(t, 2)
    if(i < size(t, 2))
      fprintf(fd, '   %d, %d, -1,\n', t(1, i) - 1, t(2, i) - 1);
      fprintf(fd, '   %d, %d, -1,\n', t(2, i) - 1, t(3, i) - 1);
      fprintf(fd, '   %d, %d, -1,\n', t(3, i) - 1, t(1, i) - 1);
    else
      fprintf(fd, '   %d, %d, -1,\n', t(1, i) - 1, t(2, i) - 1);
      fprintf(fd, '   %d, %d, -1,\n', t(2, i) - 1, t(3, i) - 1);
      fprintf(fd, '   %d, %d, -1\n', t(3, i) - 1, t(1, i) - 1);
    end
  end

  fprintf(fd, '  ]\n');
  fprintf(fd, ' }\n');



  fprintf(fd, '}\n');

  fclose(fd);

  %cmd = sprintf('SceneViewer %s', fname);
  %system(cmd, 1, 'async');

  %sleep(1);

  %unlink(fname);
