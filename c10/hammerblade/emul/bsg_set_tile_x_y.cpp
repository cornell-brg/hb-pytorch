#include <bsg_set_tile_x_y.h>

thread_local int __bsg_x;                   //The X Cord inside a tile group
thread_local int __bsg_y;                   //The Y Cord inside a tile group
thread_local int __bsg_id;                  //The ID of a tile in tile group
thread_local int __bsg_grp_org_x = 0;       //The X Cord of the tile group origin
thread_local int __bsg_grp_org_y = 0;       //The Y Cord of the tile group origin
thread_local int __bsg_grid_dim_x = 1;      //The X Dimensions of the grid of tile groups
thread_local int __bsg_grid_dim_y = 1;      //The Y Dimensions of the grid of tile groups
thread_local int __bsg_tile_group_id_x = 0; //The X Cord of the tile group within the grid
thread_local int __bsg_tile_group_id_y = 0; //The Y Cord of the tile group within the grid
thread_local int __bsg_tile_group_id = 0;   //The flat ID of the tile group within the grid


void bsg_set_tile_x_y() {
  return;
}
