//====================================================================
// bsg_tile_group_barrier.h
// 02/28/2020, Lin Cheng (lc873@cornell.edu)
//====================================================================
// This is an emulation of bsg_tile_group_barrier with pthread barrier

#ifndef  BSG_TILE_GROUP_BARRIER_H_
#define  BSG_TILE_GROUP_BARRIER_H_

#ifndef  BSG_TILE_GROUP_X_DIM
#error   Please define BSG_TILE_GROUP_X_DIM before including bsg_tile_group_barrier.h
#endif

#ifndef  BSG_TILE_GROUP_Y_DIM
#error   Please define BSG_TILE_GROUP_Y_DIM before including bsg_tile_group_barrier.h
#endif

typedef struct _bsg_row_barrier_ {
    unsigned char    _x_cord_start;
    unsigned char    _x_cord_end;
    unsigned char    _done_list[ BSG_TILE_GROUP_X_DIM ];
    unsigned int     _local_alert;
} bsg_row_barrier;

typedef struct _bsg_col_barrier_ {
    unsigned char    _y_cord_start;
    unsigned char    _y_cord_end;
    unsigned char    _done_list[ BSG_TILE_GROUP_Y_DIM];
    unsigned int     _local_alert;
} bsg_col_barrier;

//initial value of the bsg_barrier
#define INIT_TILE_GROUP_BARRIER( ROW_BARRIER_NAME, COL_BARRIER_NAME, x_cord_start, x_cord_end, y_cord_start, y_cord_end)\
bsg_row_barrier ROW_BARRIER_NAME = {                                                                                    \
    (x_cord_start),                                                                                                     \
    (x_cord_end),                                                                                                       \
    {0},                                                                                                                \
    0                                                                                                                   \
};                                                                                                                      \
bsg_col_barrier COL_BARRIER_NAME = {                                                                                    \
    (y_cord_start),                                                                                                     \
    (y_cord_end),                                                                                                       \
    {0},                                                                                                                \
    0                                                                                                                   \
};

//------------------------------------------------------------------
//  The main sync funciton
//------------------------------------------------------------------
void bsg_tile_group_barrier(bsg_row_barrier *p_row_b, bsg_col_barrier * p_col_b) {
// do nothing for now
}

#endif
