#include <torch/extension.h>
#include <vector>
#include <string.h>
#include <cstdlib>
#include <map>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>


#define min(x, y) (((x) < (y))? (x) : (y))

void fill_edgeToRow_cuda(int* edgeToRow, int *nodePointer, int num_nodes);
void fill_window_cuda(int* edgeToColumn, int* blockPartition, int* nodePointer,
                      int* edgeList, int blockSize_h, int blockSize_w, int num_nodes);


torch::Tensor SAG_cuda(
    torch::Tensor input,
    torch::Tensor row_pointers,
    torch::Tensor column_index
);


std::vector<torch::Tensor> cusparse_spmm_forward_cuda(
      torch::Tensor nodePointer,
    torch::Tensor edgeList,
              int num_nodes,
              int num_edges,
              int embedding_dim,
    torch::Tensor input
  );

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
  );


std::vector<torch::Tensor> cusparse_sddmm_forward_cuda(
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
              int num_nodes,
              int num_edges,
              int embedding_dim,
    torch::Tensor input
);

std::vector<torch::Tensor> spmmAGNN_forward_cuda(
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor edgeAttention,        // *edge attention [n_head, n_e]
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
              int num_nodes,
              int num_edges,
              int embedding_dim,
    torch::Tensor input
    ); 

std::vector<torch::Tensor> sddmm_forward_cuda(
    torch::Tensor nodePointer,
    torch::Tensor edgeList,			    // edge list.
    torch::Tensor blockPartition,		// number of TC_blocks (16x8) in each row_window.
    torch::Tensor edgeToColumn, 		// eid -> col within each row_window.
    torch::Tensor edgeToRow, 			  // eid -> col within each row_window.
              int num_nodes,
              int num_edges,
              int embedding_dim,	    // embedding dimension.
	  torch::Tensor input				        // input feature matrix.
); 

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor SAG(
    torch::Tensor input,
    torch::Tensor row_pointers,
    torch::Tensor column_index) 
{
  CHECK_INPUT(input);
  CHECK_INPUT(row_pointers);
  CHECK_INPUT(column_index);

  return SAG_cuda(input, row_pointers, column_index);
}

//////////////////////////////////////////
// cuSPSPMM Foward Pass (GCN, GraphSAGE)
////////////////////////////////////////////
std::vector<torch::Tensor> cusparse_spmm_forward(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList
) {
  CHECK_INPUT(input);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);

  return cusparse_spmm_forward_cuda(nodePointer, edgeList, \
                                    num_nodes, num_edges, embedding_dim, \
                                    input);
}


//////////////////////////////////////////
// SPMM Foward Pass (GCN, GraphSAGE)
////////////////////////////////////////////
std::vector<torch::Tensor> spmm_forward(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow
) {
  CHECK_INPUT(input);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(blockPartition);
  CHECK_INPUT(edgeToColumn);
  CHECK_INPUT(edgeToRow);

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);

  return spmm_forward_cuda(nodePointer, edgeList, 
                            blockPartition, edgeToColumn, edgeToRow, 
                            num_nodes, num_edges, embedding_dim,
                            input);
}

////////////////////////////////////////////
//
// SPMM Foward Pass (AGNN, AGNN)
//
////////////////////////////////////////////
std::vector<torch::Tensor> spmm_forward_AGNN(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor edgeAttention,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow
) {
  CHECK_INPUT(input);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(edgeAttention);
  CHECK_INPUT(blockPartition);
  CHECK_INPUT(edgeToColumn);
  CHECK_INPUT(edgeToRow);

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);
  
  return spmmAGNN_forward_cuda(nodePointer, edgeList, edgeAttention,
                              blockPartition, edgeToColumn, edgeToRow, 
                              num_nodes, num_edges, embedding_dim,
                              input);
}



//////////////////////////////////////////
// cuSPSPMM Foward Pass (GCN, GraphSAGE)
////////////////////////////////////////////
std::vector<torch::Tensor> cusparse_sddmm_forward(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList
) {
  CHECK_INPUT(input);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);

  return cusparse_sddmm_forward_cuda(nodePointer, edgeList, \
                                    num_nodes, num_edges, embedding_dim, \
                                    input);
}

////////////////////////////////////////////
//
// SDDMM Foward Pass
//
////////////////////////////////////////////
std::vector<torch::Tensor> sddmm_forward(
  torch::Tensor input,				
  torch::Tensor nodePointer,
  torch::Tensor edgeList,			    
	torch::Tensor blockPartition,		
	torch::Tensor edgeToColumn, 		
	torch::Tensor edgeToRow
) {
  CHECK_INPUT(input);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(blockPartition);
  CHECK_INPUT(edgeToColumn);
  CHECK_INPUT(edgeToRow);

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);

//   printf("at sddmm_forward\n");
  return sddmm_forward_cuda(nodePointer, edgeList, 
                            blockPartition, edgeToColumn, edgeToRow, 
                            num_nodes, num_edges, embedding_dim,
                            input);
}


// condense an sorted array with duplication: [1,2,2,3,4,5,5]
// after condense, it becomes: [1,2,3,4,5].
// Also, mapping the origin value to the corresponding new location in the new array.
// 1->[0], 2->[1], 3->[2], 4->[3], 5->[4]. 
std::map<unsigned, unsigned> inplace_deduplication(unsigned* array, unsigned length){
    int loc=0, cur=1;
    std::map<unsigned, unsigned> nb2col;
    nb2col[array[0]] = 0;
    while (cur < length){
        if(array[cur] != array[cur - 1]){
            loc++;
            array[loc] = array[cur];
            nb2col[array[cur]] = loc;       // mapping from eid to TC_block column index.[]
        }
        cur++;
    }
    return nb2col;
}

void preprocess(torch::Tensor edgeList_tensor, 
                torch::Tensor nodePointer_tensor, 
                int num_nodes, 
                int blockSize_h,
                int blockSize_w,
                torch::Tensor blockPartition_tensor, 
                torch::Tensor edgeToColumn_tensor,
                torch::Tensor edgeToRow_tensor
                ){

    // input tensors.
    auto edgeList = edgeList_tensor.accessor<int, 1>();
    auto nodePointer = nodePointer_tensor.accessor<int, 1>();

    // output tensors.
    auto blockPartition = blockPartition_tensor.accessor<int, 1>();
    auto edgeToColumn = edgeToColumn_tensor.accessor<int, 1>();
    auto edgeToRow = edgeToRow_tensor.accessor<int, 1>();

    unsigned block_counter = 0;

    #pragma omp parallel for 
    for (unsigned nid = 0; nid < num_nodes; nid++){
        for (unsigned eid = nodePointer[nid]; eid < nodePointer[nid+1]; eid++)
            edgeToRow[eid] = nid;
    }

    #pragma omp parallel for reduction(+:block_counter)
    for (unsigned iter = 0; iter < num_nodes + 1; iter +=  blockSize_h){
        unsigned windowId = iter / blockSize_h;
        unsigned block_start = nodePointer[iter];
        unsigned block_end = nodePointer[min(iter + blockSize_h, num_nodes)];
        unsigned num_window_edges = block_end - block_start;
        unsigned *neighbor_window = (unsigned *) malloc (num_window_edges * sizeof(unsigned));
        memcpy(neighbor_window, &edgeList[block_start], num_window_edges * sizeof(unsigned));

        // Step-1: Sort the neighbor id array of a row window.
        thrust::sort(neighbor_window, neighbor_window + num_window_edges);

        // Step-2: Deduplication of the edge id array.
        // printf("Before dedupblication: %d\n", num_window_edges);
        std::map<unsigned, unsigned> clean_edges2col = inplace_deduplication(neighbor_window, num_window_edges);

        // generate blockPartition --> number of TC_blcok in each row window.
        blockPartition[windowId] = (clean_edges2col.size() + blockSize_w - 1) /blockSize_w;
        block_counter += blockPartition[windowId];

        // scan the array and generate edge to column mapping. --> edge_id to compressed_column_id of TC_block.
        for (unsigned e_index = block_start; e_index < block_end; e_index++){
            unsigned eid = edgeList[e_index];
            edgeToColumn[e_index] = clean_edges2col[eid];
        }
    }
    printf("TC_Blocks:\t%d\nExp_Edges:\t%d\n", block_counter, block_counter * 8 * 16);
}


void preprocess_gpu(torch::Tensor edgeList_tensor, 
                torch::Tensor nodePointer_tensor, 
                int num_nodes, 
                int blockSize_h,
                int blockSize_w,
                torch::Tensor blockPartition_tensor, 
                torch::Tensor edgeToColumn_tensor,
                torch::Tensor edgeToRow_tensor
                )
{

    // input tensors.
    auto edgeList = edgeList_tensor.data<int>();
    auto nodePointer = nodePointer_tensor.data<int>();

    // output tensors.
    auto blockPartition = blockPartition_tensor.data<int>();
    auto edgeToColumn = edgeToColumn_tensor.data<int>();
    auto edgeToRow = edgeToRow_tensor.data<int>();

    unsigned block_counter = 0;
    
    fill_edgeToRow_cuda(edgeToRow, nodePointer, num_nodes);
    fill_window_cuda(edgeToColumn, blockPartition, nodePointer, edgeList,
                                blockSize_h, blockSize_w, num_nodes);

    printf("TC_Blocks:\t%d\nExp_Edges:\t%d\n", block_counter, block_counter * 8 * 16);
}

torch::Tensor SpMM_validate(
    torch::Tensor input,
    torch::Tensor row_pointers,
    torch::Tensor column_index
){  

  torch::Device device(torch::kCPU);
  torch::Tensor output_cpu = torch::zeros_like(input).to(device);
  torch::Tensor input_cpu = input.to(device);
  torch::Tensor row_pointers_cpu= row_pointers.to(device);
  torch::Tensor column_index_cpu = column_index.to(device);

  auto output_cpu_acc = output_cpu.accessor<float,2>();
  auto input_cpu_acc = input_cpu.accessor<float,2>();
  auto row_pointers_cpu_acc = row_pointers_cpu.accessor<int,1>();
  auto column_index_cpu_acc = column_index_cpu.accessor<int,1>();

  // Iterate all nodes. 
  for (int s_nid = 0; s_nid < input_cpu.size(0); s_nid++){
    int nb_begin = row_pointers_cpu_acc[s_nid];
    int nb_end = row_pointers_cpu_acc[s_nid + 1];

    for (int nb_idx = nb_begin; nb_idx < nb_end; nb_idx++){
      int nid = column_index_cpu_acc[nb_idx];
      for (int d = 0; d < input_cpu.size(1); d++){
        output_cpu_acc[s_nid][d] +=input_cpu_acc[nid][d];
      }
    }
  }
  return output_cpu;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess", &preprocess, "Preprocess Step (CPU)");
  m.def("preprocess_gpu", &preprocess_gpu, "Preprocess Step (CUDA)");

  // forward computation
  m.def("SAG", &SAG, "GNNAdvisor base Scatter-and-Gather Kernel (CUDA)");
  m.def("forward", &spmm_forward, "TC-GNN SPMM forward (CUDA)");
  m.def("cusparse_spmm", &cusparse_spmm_forward, "cuSPARSE SpMM (CUDA)");
  m.def("SpMM_validate", &SpMM_validate, "SpMM validate kernel on (CPU)");

  m.def("forward_ef", &sddmm_forward, "TC-GNN SDDMM forward (CUDA)");
  m.def("forward_AGNN", &spmm_forward_AGNN, "TC-GNN SPMM (AGNN) forward (CUDA)");
  m.def("cusparse_sddmm", &cusparse_sddmm_forward, "cuSPARSE SDDMM (CUDA)");

  // backward
  m.def("backward", &spmm_forward, "TC-GNN SPMM backward (CUDA)");
  m.def("backward_ef", &sddmm_forward, "TC-GNN SDDMM backward_ef (CUDA)");
}