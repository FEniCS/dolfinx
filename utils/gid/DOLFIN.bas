*#FILE_EXTENSION .xml
*realformat "%-f"
<?xml version="1.0"?>
<dolfin xmlns:dolfin="http://fenicsproject.org">
  <mesh celltype="tetrahedron" dim="3">
    <vertices size="*npoin">
*loop nodes
      <vertex index="*operation(nodesnum-1)" x="*nodescoord(1)" y="*nodescoord(2)" z="*nodescoord(3)"/>
*end nodes
    </vertices>
    <cells size="*nelem">
*loop elems
      <tetrahedron index="*operation(elemsnum-1)" v0="*operation(elemsconec(1)-1)" v1="*operation(elemsconec(2)-1)" v2="*operation(elemsconec(3)-1)" v3="*operation(elemsconec(4)-1)" />
*end elems
    </cells>
  </mesh>
</dolfin>
