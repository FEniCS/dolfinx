// Persistence Of Vision raytracer version 3.1 sample file.

global_settings { assumed_gamma 2.2 }

//#include "colors.inc"           // Standard colors library

//camera {
//   location  <1.0, 1.0, 5.0>
//   direction <0, 0,    -1>
//   up        <0, 1,    0>
//   right   <1, 0,    0>
//   look_at   <0.5, 0.5,    0>
//}

camera {
   location  <1.5, 0.0, 6.0>
   direction <0, 0,    -1>
   up        <0, 1,    0>
   right   <1, 0,    0>
   look_at   <1.0, -0.5,    0>
}

//camera {
//   location  <2.8, 0.0, 7.0>
//   direction <0, 0,    -1>
//   up        <0, 1,    0>
//   right   <1, 0,    0>
//   look_at   <2.8, -1.0,    0>
//}


// Light source

light_source {< -20, 21, 20>  color <1,1,1>  }
light_source {< 1, 1, 20>  color <1,1,1>  }
light_source {< 20, -10, 20>  color <1,1,1> }

//#declare sphereTexture =
// texture {
//  pigment { color red 0.8 green 0.8 blue 0.8 filter 0.6}
// }


//plane { z, -10
//  pigment { color Gray }
//}


