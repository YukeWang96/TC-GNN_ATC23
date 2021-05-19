#!/usr/bin/env python3
from collections import defaultdict
import sys 
import math

dense_tile_H = 8
dense_tile_W = 8

dataset = [
		# ('toy'	        , 3	    , 2   ),  
		# ('tc_gnn_verify'	, 16	, 2),
		# ('tc_gnn_verify_2x'	, 16	, 2),

		# ('citeseer'	        		, 3703	    , 6   ),  
		# ('cora' 	        		, 1433	    , 7   ),  
		# ('pubmed'	        		, 500	    , 3   ),      
		# ('ppi'	            		, 50	    , 121 ),   
		
		# ('PROTEINS_full'             , 29       , 2) ,   
		# ('OVCAR-8H'                  , 66       , 2) , 
		# ('Yeast'                     , 74       , 2) ,
		# ('DD'                        , 89       , 2) ,
		# ('YeastH'                    , 75       , 2) ,   
		# ('SW-620H'                   , 66       , 2) ,

		( 'amazon0505'               , 96	  , 22),
		( 'artist'                   , 100	  , 12),
		( 'com-amazon'               , 96	  , 22),
		( 'soc-BlogCatalog'	         , 128	  , 39),      
		( 'amazon0601'  	         , 96	  , 22), 


		# ( 'web-BerkStan'             , 100	  , 12),
		# ( 'Reddit'                   , 602    , 41),

		# ( 'wiki-topcats'             , 300	  , 12),
		# ( 'COLLAB'                   , 100      , 3) ,
		# ( 'wiki-topcats'             , 300	  , 12),
		# ( 'Reddit'                   , 602      , 41),
		# ( 'enwiki-2013'	           , 100	  , 12),      
		# ( 'amazon_also_bought'       , 96       , 22),
]


data_dir = '/home/yuke/.graphs/orig/'
# print(data_dir)
# print("dataset,origin,origin_eff,reduced,reduced_eff,reduction (%)")

def find_dense(path, data):
	fp = open(path)
	nodes = set()

	graph = defaultdict(list)
	for line in fp:
		src, dst = line.strip('\n').split(" ")
		src, dst = int(src), int(dst)
		nodes.add(src)
		nodes.add(dst)
		graph[dst].append(src)
	num_nodes = max(nodes)


	# blk_H = math.ceil(num_nodes/dense_tile_H)
	# blk_W = math.ceil(num_nodes/dense_tile_W)

	# print(blk_H * blk_W)
	# tiles = [0] * (blk_H * blk_W)

	# for src, dst in edges:
	# 	blk_id_H = math.floor(src/dense_tile_H)
	# 	blk_id_W = math.floor(dst/dense_tile_W)
	# 	global_blk_idx = blk_id_H * blk_W + blk_id_W
	# 	tiles[global_blk_idx] += 1
	tile_cnt = 0
	opt_cnt = 0
	chunk_edges = []
	for src_iter in range(0, num_nodes, dense_tile_H):

		dst_list = []
		for src in range(src_iter, src_iter + dense_tile_H):
			dst_list += graph[src]

		actual_cnt = len(dst_list)
		chunk_edges.append(len(dst_list))

		range_set = sorted(list(set(dst_list)))

		# TC-GNN tiles
		opt_cnt += (len(range_set) + dense_tile_W - 1)//dense_tile_W
		tmp_opt_cnt = (len(range_set) + dense_tile_W - 1)//dense_tile_W
		exp_opt_cnt = (dense_tile_H * dense_tile_W) * tmp_opt_cnt


		# naive sliding window without compression.
		tmp = 0
		range_set = sorted(list(range_set))
		i = j = 0
		while i < len(range_set) and j < len(range_set):
			end = range_set[i] + dense_tile_W
			while j < len(range_set) and range_set[j] < end:
				j += 1
			i = j
			tile_cnt += 1
			tmp += 1

		exp_tile_cnt =  (dense_tile_H * dense_tile_W) * tile_cnt

		if tmp < tmp_opt_cnt:
			print(range_set)
			print(tmp, tmp_opt_cnt)
			print("tmp < tmp_opt_cnt Error Encounter, Duplicate Edges")
			sys.exit(0)

	# print("{:10},Avg.Chunk.Size: {:.2f}".format(data, np.mean(chunk_edges)))
	# print("{},{},{:.2f},{},{:.2f},{:.2f}".format(data, tile_cnt, \
	# 											actual_cnt/exp_tile_cnt, \
	# 											opt_cnt, actual_cnt/exp_opt_cnt,  \
	# 											100 * (tile_cnt - opt_cnt) / tile_cnt))

	naive_blockPerRow = math.ceil(tile_cnt/(num_nodes//dense_tile_H))
	tcgnn_blockPerRow = math.ceil(opt_cnt/(num_nodes//dense_tile_H))
	print("{},{},{},".format(data, naive_blockPerRow, tcgnn_blockPerRow, math.ceil(num_nodes//dense_tile_H)))

	# plt.hist(tiles, bins=100)
	# plt.savefig("{}.pdf".format(data))
	# print(Counter(tiles))
	# return tiles
if __name__ == '__main__':
	print("Dataset,Naive BPW,TC-GNN BPW, Total W")
	for data, d, c in dataset:
		find_dense(data_dir + data, data)