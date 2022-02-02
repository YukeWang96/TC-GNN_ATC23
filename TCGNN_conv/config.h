#ifndef CONFIG_H
#define CONFIG_H

// for wmma::precision::tf32
#define BLK_H 16 
#define BLK_W 8
// for half
// #define BLK_H 16 
// #define BLK_W 16

#define WARP_SIZE 32
// #define WPB 1 		// --> MAX_DIM:	16 = 16 * 1
// #define WPB 2 	// --> MAX_DIM: 32 = 16 * 2
#define WPB 4 	// --> MAX_DIM: 64 = 16 * 4
// #define WPB 8 	// --> MAX_DIM: 128 = 16 * 8
// #define WPB 16	// --> MAX_DIM: 256 = 16 * 16
// #define WPB 32	// --> MAX_DIM: 512 = 32 * 16

// #define verify
#define SPMM
// #define SDDMM

#endif