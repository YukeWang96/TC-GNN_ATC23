#include <torch/extension.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <mma.h>
#include <cuda_runtime.h>

#include "config.h"

using namespace nvcuda;

#define MAX_DIM 100
#define MAX_NB 100       // must <= partsize 
#define threadPerWarp 2 //must < 32
#define wrapPerBlock 1  // must also set with respect to the 
                        // [thread-per-block = wrapPerBlock *  threadPerWarp]

__device__ inline float atomicAdd_F(float* address, float value)
{
  float old = value;  
  while ((old = atomicExch(address, atomicExch(address, 0.0f)+old))!=0.0f);
}

//////////////////////
/// SPMM forward
//////////////////////
__global__ void spmm_forward_cuda_kernel(
	const int * __restrict__ nodePointer,		// node pointer.
	const int *__restrict__ edgeList,			// edge list.
	const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
	const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
	const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ in_mat,		    // input feature matrix.
	float *out_mat							    // aggregated output feature matrix.
);

//////////////////////
/// SDDMM forward
//////////////////////
__global__ void sddmm_forward_cuda_kernel(
    const int * __restrict__ nodePointer,		// node pointer.
	const int *__restrict__ edgeList,			// edge list.
	const int *__restrict__ blockPartition,		// number of TC_blocks (16x8) in each row_window.
	const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
	const int *__restrict__ edgeToRow, 			// eid -> col within each row_window.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *__restrict__ in_mat,					// input feature matrix.
	float *edgeFeature							// aggregated output feature matrix.
);


template <typename scalar_t>
__global__ void spmm_backward_cuda_kernel(
    int num_nodes, 
    int dim,
    int num_parts,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_input,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> column_index,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> degrees,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part2Node
);


////////////////////////////////////////////
//
// SPMM Foward Pass
//
////////////////////////////////////////////
std::vector<torch::Tensor> spmm_forward_cuda(
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
              int num_nodes,
              int num_edges,
              int embedding_dim,
    torch::Tensor input
) 
{
    auto output = torch::zeros_like(input);
    const int num_row_windows = blockPartition.size(0);
    const int WARPperBlock = 16;

    dim3 grid(num_row_windows, 1, 1);
    dim3 block(WARP_SIZE, WARPperBlock, 1);

    const int dimTileNum = embedding_dim / BLK_H;
	const int dynamic_shared_size = dimTileNum * BLK_W * BLK_H * sizeof(float); // dynamic shared memory.

    spmm_forward_cuda_kernel<<<grid, block, dynamic_shared_size>>>(
                                                                    nodePointer.data<int>(), 
                                                                    edgeList.data<int>(),
                                                                    blockPartition.data<int>(), 
                                                                    edgeToColumn.data<int>(), 
                                                                    edgeToRow.data<int>(), 
                                                                    num_nodes,
                                                                    num_edges,
                                                                    embedding_dim,
                                                                    input.data<float>(), 
                                                                    output.data<float>()
                                                                );

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    return {output};
}

////////////////////////////////////////////
//
// SPMM Foward Pass
//
////////////////////////////////////////////
std::vector<torch::Tensor> sddmm_forward_cuda(
    torch::Tensor nodePointer,
    torch::Tensor edgeList,			    // edge list.
	torch::Tensor blockPartition,		// number of TC_blocks (16x8) in each row_window.
	torch::Tensor edgeToColumn, 		// eid -> col within each row_window.
	torch::Tensor edgeToRow, 			// eid -> col within each row_window.
              int num_nodes,
              int num_edges,
              int embedding_dim,	    // embedding dimension.
	torch::Tensor input				    // input feature matrix.
) 
{
    auto output = torch::zeros_like(edgeList);
    const int num_row_windows = blockPartition.size(0);

	dim3 grid(num_row_windows, 1, 1);
	dim3 block(WARP_SIZE, 1, 1);

    sddmm_forward_cuda_kernel<<< grid, block>>>(
                                                nodePointer.data<int>(), 
                                                edgeList.data<int>(),
                                                blockPartition.data<int>(), 
                                                edgeToColumn.data<int>(), 
                                                edgeToRow.data<int>(), 
                                                num_nodes,
                                                num_edges,
                                                embedding_dim,
                                                input.data<float>(), 
                                                output.data<float>()
                                                );

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    return {output};
}


//////////////////////
/// SPMM forward
//////////////////////
__global__ void spmm_forward_cuda_kernel(
	const int * __restrict__ nodePointer,		// node pointer.
	const int *__restrict__ edgeList,			// edge list.
	const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
	const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
	const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // aggregated output feature matrix.
) {
    const unsigned bid = blockIdx.x;								// block_index == row_window_index
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16.
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block.
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.

	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
	const unsigned nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.
	
	const unsigned eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
	const unsigned eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
	const unsigned num_TC_blocks = blockPartition[bid]; 			// number of TC_blocks of the current row_window.
	const unsigned dense_bound = numNodes * embedding_dim;

	__shared__ float sparse_A[BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[BLK_W];					// TC_block col to dense_tile row.
	// __shared__ float dense_X[dimTileNum * BLK_W * BLK_H];	// column-major dense tile [dimTileNum, BLK_W, BLK_H]
	extern __shared__ float dense_X[];

	wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
	wmma::fill_fragment(acc_frag, 0.0f);

	// Processing TC_blocks along the column dimension of Sparse A.
	for (unsigned i = 0; i < num_TC_blocks; i++){

		// Init A_colToX_row with dummy values.
		if (tid < BLK_W){
			sparse_AToX_index[tid] = numNodes + 1;
		}

		__syncthreads();

		// Init sparse_A with zero values.
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock){
			sparse_A[idx] = 0;
		}

		// Init dense_X with zero values.
		#pragma unroll
		for (unsigned idx = tid; idx < dimTileNum * BLK_W * BLK_H; idx += threadPerBlock){
			dense_X[idx] = 0;
		}

		// Initialize sparse_A by using BLK_H (16) threads from the warp-0.
		// currently fetch all neighbors of the current nodes.
		// then to see whether it can fit into current TC_block frame of column.		
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock){
			unsigned col = edgeToColumn[eIdx];
			if (i * BLK_W <= col && col < (i + 1) * BLK_W){			// if the edge in the current TC_block frame of column.
				unsigned row_local = edgeToRow[eIdx] % BLK_H;
				unsigned col_local = col % BLK_W;
				sparse_A[row_local * BLK_W + col_local] = 1;		// set the edge of the sparse_A.
				sparse_AToX_index[col_local] = edgeList[eIdx];		// record the mapping from sparse_A colId to rowId of dense_X.
			}		
		}

		__syncthreads();

		// Initialize dense_X by column-major store,
		// Threads of a warp for fetching a dense_X.
		// each warp identify by wid.
		if (wid < dimTileNum)
			#pragma unroll
			for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += warpSize){
				unsigned dense_rowIdx = sparse_AToX_index[idx % BLK_W];						// TC_block_col to dense_tile_row.
				unsigned dense_dimIdx = idx / BLK_W;										// dimIndex of the dense tile.
				unsigned source_idx = dense_rowIdx * embedding_dim + wid * BLK_H + dense_dimIdx;
				unsigned target_idx = wid * BLK_W * BLK_H + idx;
				// boundary test.
				if (source_idx >= dense_bound)
					dense_X[target_idx] = 0;
				else
					dense_X[target_idx] = input[source_idx];
			}

		__syncthreads();

		if (wid < dimTileNum)
		{
			wmma::load_matrix_sync(a_frag, sparse_A, BLK_W);
			wmma::load_matrix_sync(b_frag, dense_X + wid * BLK_W * BLK_H, BLK_W);

			#pragma unroll
			for (unsigned t = 0; t < a_frag.num_elements; t++) {
				a_frag.x[t] =  wmma::__float_to_tf32(a_frag.x[t]);
			}

			#pragma unroll
			for (unsigned t = 0; t < b_frag.num_elements; t++) {
				b_frag.x[t] =  wmma::__float_to_tf32(b_frag.x[t]);
			}
			// Perform the matrix multiplication.
			wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
		}
	}

	if (wid < dimTileNum)
		// Store the matrix to output matrix.
		// * Note * embeeding dimension should be padded divisible by BLK_H for output correctness.
		wmma::store_matrix_sync(output + bid * BLK_H * embedding_dim + wid * BLK_H, acc_frag, embedding_dim, wmma::mem_row_major);
}


//////////////////////
/// SDDMM forward
//////////////////////
__global__ void sddmm_forward_cuda_kernel(
    const int *__restrict__ nodePointer,
	const int *__restrict__ edgeList,			// edge list.
	const int *__restrict__ blockPartition,		// number of TC_blocks (16x8) in each row_window.
	const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
	const int *__restrict__ edgeToRow, 			// eid -> col within each row_window.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *__restrict__ in_mat,					// input feature matrix.
	float *edgeFeature							// aggregated output feature matrix.
)
{
    unsigned bid = blockIdx.x;										// block_index == row_window_index
    unsigned wid = threadIdx.y;										// warp_index handling multi-dimension > 16.
    unsigned laneid = threadIdx.x;									// lanid of each warp.
    unsigned tid = threadIdx.y * blockDim.x + laneid;				// threadid of each block.

    unsigned threadPerBlock = blockDim.x * blockDim.y;
    unsigned DimIterations =  (embedding_dim + BLK_W - 1) / BLK_W; 	// dimension iteration for output.

    unsigned nid_start = bid * BLK_H;								// starting node_id of current row_window.
    unsigned nid_end = min((bid + 1) * BLK_H, numNodes);			// ending node_id of the current row_window.

    unsigned eIdx_start = nodePointer[nid_start];					            // starting eIdx of current row_window.
    unsigned eIdx_end = nodePointer[nid_end];						            // ending eIdx of the current row_window.
    unsigned num_TC_blocks = (blockPartition[bid] * BLK_W + BLK_H - 1)/BLK_H; 	// number of TC_blocks of the current row_window.

    __shared__ float sparse_A[BLK_H * BLK_H];					// 16 x 16 output sparse matrix.
    __shared__ float sparse_A_val[BLK_H * BLK_H];				// 16 x 16 output sparse matrix.
    #ifdef verify
    __shared__ float verify_A[BLK_H * BLK_H];					// 16 x 16 output sparse matrix.
    #endif 

    __shared__ unsigned sparse_AToX_index[BLK_H];				// TC_block col to dense_tile row.
    __shared__ float dense_X[BLK_H * BLK_W];
    __shared__ float dense_Y[BLK_W * BLK_H];

    wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // Processing TC_blocks along the column dimension of Sparse A.
    // The block step here is 2, which is 16 = 8 + 8. 
    // In order to reuse the edgeToColumn in SpMM. 
    for (unsigned i = 0; i < num_TC_blocks; i++ ){

        if (wid == 0 && laneid < BLK_H){
            sparse_AToX_index[laneid] = numNodes + 1;
        }

        __syncthreads();

        #pragma unroll
        for (unsigned idx = tid; idx < BLK_H * BLK_H; idx += threadPerBlock){
            sparse_A[idx] = numEdges + 1;
            sparse_A_val[idx] = 0.0f;
        }

        #pragma unroll
        for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock){
            dense_X[idx] = 0;
            dense_Y[idx] = 0;
        }

        // Initialize sparse_A by using BLK_H (16) threads from the warp-0.
        // currently fetch all neighbors of the current nodes.
        // then to see whether it can fit into current TC_block frame of column.
        #pragma unroll
        // if (tid < WARP_SI)
        for (unsigned eIdx = tid + eIdx_start; eIdx < eIdx_end; eIdx += threadPerBlock){
            unsigned col = edgeToColumn[eIdx];						// condensed column id in sparse_A.
            if (i * BLK_H <= col && col < (i + 1) * BLK_H){			// if the edge in the current TC_block frame of column.
                unsigned row = edgeToRow[eIdx] % BLK_H;				// reverse indexing the row Id of the edge.
                sparse_A[row * BLK_H + col % BLK_H] = eIdx;			// set the edge of the sparse_A.
                sparse_AToX_index[col % BLK_H] = edgeList[eIdx];	// record the mapping from sparse_A colId to rowId of dense_X.
            }
        }		

        __syncthreads();

        for (unsigned warp_iter = 0; warp_iter < DimIterations; warp_iter++){
            // Initialize dense_X by row-major store,
            // Threads of a warp for fetching a dense_X.
            #pragma unroll
            for (unsigned i = tid; i < BLK_H * BLK_W; i += threadPerBlock){
                unsigned dense_rowIdx = i / BLK_W;					
                unsigned dense_dimIdx = i % BLK_W;					
                unsigned target_idx = i;
                unsigned source_idx = (nid_start + dense_rowIdx) * embedding_dim + warp_iter * BLK_W + dense_dimIdx;
                if (source_idx >= numNodes * embedding_dim)
                    dense_X[target_idx] = 0;
                else
                    dense_X[target_idx] = in_mat[source_idx];
            }

            // Initialize dense_Y by column-major store,
            // Threads of a warp for fetching a dense_Y.
            #pragma unroll
            for (unsigned i = tid; i < BLK_W * BLK_H; i += threadPerBlock){
                unsigned dense_rowIdx = sparse_AToX_index[i / BLK_W];					// TC_block_col to dense_tile_row.
                unsigned dense_dimIdx = i % BLK_W;										// dimIndex of the dense tile.
                unsigned target_idx = i;
                unsigned source_idx = dense_rowIdx * embedding_dim + warp_iter * BLK_W + dense_dimIdx;
                if (source_idx >= numNodes * embedding_dim)
                    dense_Y[target_idx] = 0;
                else
                    dense_Y[target_idx] = in_mat[source_idx];
            }

            __syncthreads();

            wmma::load_matrix_sync(a_frag, dense_X, BLK_W);
            wmma::load_matrix_sync(b_frag, dense_Y, BLK_W);

            #pragma unroll
            for (unsigned t = 0; t < a_frag.num_elements; t++) {
                a_frag.x[t] =  wmma::__float_to_tf32(a_frag.x[t]);
            }

            #pragma unroll
            for (unsigned t = 0; t < b_frag.num_elements; t++) {
                b_frag.x[t] =  wmma::__float_to_tf32(b_frag.x[t]);
            }

            // Perform the matrix multiplication on Tensor Core
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);		
        } // <--- ending of warp iteration.

        wmma::store_matrix_sync(sparse_A_val, acc_frag, BLK_H, wmma::mem_row_major);
        wmma::fill_fragment(acc_frag, 0.0f);

        // Output the results to sparse matrix edge featureList.
        for (unsigned t = 0; t < BLK_H * BLK_H; t++) {
            unsigned rowId = t / BLK_H;
            unsigned colId = t % BLK_H;
            if (sparse_A[rowId * BLK_H + colId] < numEdges){
                unsigned eIdx = sparse_A[rowId * BLK_H + colId];
                edgeFeature[eIdx] = sparse_A_val[t];
            }
        } //<-- ending of storing output to global memory.
    }
}

////////////////////////////////////////////
// 
// backward pass
//
////////////////////////////////////////////
std::vector<torch::Tensor> spmm_backward_cuda(
    int threadPerBlock,
    torch::Tensor d_output,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node
) {

    auto d_input = torch::zeros_like(d_output);
    // d_input = d_output;
    const int dim = d_input.size(1);
    const int num_nodes = d_input.size(0);
    const int num_parts = part2Node.size(0);
    const int blocks = (num_parts * 32 + threadPerBlock - 1) / threadPerBlock; 

    AT_DISPATCH_FLOATING_TYPES(d_output.type(), "spmm_cuda_backward", ([&] {
                                spmm_backward_cuda_kernel<scalar_t><<<blocks, threadPerBlock>>>(
                                    num_nodes, 
                                    dim,
                                    num_parts,
                                    d_output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    d_input.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    row_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    column_index.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    degrees.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                                    part_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(), 
                                    part2Node.packed_accessor32<int,1,torch::RestrictPtrTraits>()
                                );
                            }));
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    return {d_input};
}

template <typename scalar_t>
__global__ void spmm_backward_cuda_kernel(
    int num_nodes, 
    int dim,
    int num_parts, 
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_input,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> column_index,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> degrees,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part2Node
) {

    int tid =  blockIdx.x * blockDim.x + threadIdx.x;
    int warpId =  tid / 32;
    int intraWarp_tid = tid % 32;
    int block_warpID = threadIdx.x/32;
    
    if (warpId < num_parts && intraWarp_tid < threadPerWarp){

        __shared__  int partial_index[MAX_NB * wrapPerBlock];
        __shared__ float partial_results[MAX_DIM * wrapPerBlock];

        int srcId = part2Node[warpId];
        int partBeg = part_pointers[warpId];
        int partEnd = part_pointers[warpId + 1];
        float src_norm = degrees[srcId];

        int pindex_base = block_warpID * MAX_NB;
        for (int nid = partBeg + intraWarp_tid; nid < partEnd; nid += threadPerWarp){
            partial_index[pindex_base + nid - partBeg] = column_index[nid];
        }
         __syncthreads();

        int presult_base = block_warpID * MAX_DIM;
        for (int nid = 0; nid < partEnd - partBeg; nid++)
        {
            int nIndex = partial_index[pindex_base + nid];
            float degree_norm =  __fmaf_rn(src_norm, degrees[nIndex], 0);

            if (nid == 0)
                for (int d = intraWarp_tid; d < dim; d += threadPerWarp){
                    partial_results[presult_base + d] = 0;
                    // atomicAdd_F((float*)&d_input[srcId][d], degree_norm * d_output[nIndex][d]);
                }
            for (int d = intraWarp_tid; d < dim; d += threadPerWarp){
                partial_results[presult_base + d] += __fmaf_rn(degree_norm, d_output[nIndex][d], 0);
            }
        }
        for (int d = intraWarp_tid; d < dim; d += threadPerWarp){
            atomicAdd_F((float*)&d_input[srcId][d], partial_results[presult_base + d]);
        }
    }
}
