*realformat "%1.5f"
*intformat "%i"
*# Interface GID-Dolfin 
*# To export mesh of triangles in 2D and 3D 
*# Copyright (C) 2004 Harald Svensson.
*# Licensed under the GNU GPL Version 2.
<?xml version="1.0" encoding="UTF-8"?>
<dolfin xmlns:dolfin="http://www.phi.chalmers.se/dolfin/">
 <mesh>
*Set var HANDLE=0
*loop nodes
*Set var HANDLE=operation(HANDLE(int)+1)
*End nodes
  <nodes size="*HANDLE">
*loop nodes    
   <node name="*NodesNum" x="*NodesCoord(1,real)" y="*NodesCoord(2,real)" z="*NodesCoord(3,real)"/>
*End nodes
  </nodes>
*Set elems(Triangle)
  <cells size="*nelem(Triangle)">
*loop elems
    <triangle name="*ElemsNum" n0="*operation(elemsConec(1,Int)-1)" n1="*operation(elemsConec(2,int)-1)" n2="*operation(elemsConec(3,int)-1)"/>
*End elems
  </cells>
*Set elems(Tetrahedra)
  <cells size="*nelem(Tetrahedra)">
*loop elems
    <tetrahedra name="*ElemsNum" n0="*operation(elemsConec(1,Int)-1)" n1="*operation(elemsConec(2,int)-1)" n2="*operation(elemsConec(3,int)-1) n3="*operation(elemsConec(4,int)-1)"/>
*End elems
  </cells>
 </mesh>
</dolfin>
