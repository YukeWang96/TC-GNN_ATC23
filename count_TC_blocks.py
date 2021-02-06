#!/usr/bin/env python3
import subprocess
import datetime
import os
# import matplotlib.pyplot as plt
import math
from collections import Counter, defaultdict
import sys 

dense_tile_H = 16
dense_tile_W = 8

dataset = [
		# ('toy'	        , 3	    , 2   ),  
		# ('tc_gnn_verify'	, 16	, 2),
		# ('tc_gnn_verify_2x'	, 16	, 2),

		# ('citeseer'	        		, 3703	    , 6   ),  
		('cora' 	        		, 1433	    , 7   ),  
		('pubmed'	        		, 500	    , 3   ),      
		('ppi'	            		, 50	    , 121 ),   
		
		('PROTEINS_full'             , 29       , 2) ,   
		('OVCAR-8H'                  , 66       , 2) , 
		('Yeast'                     , 74       , 2) ,
		('DD'                        , 89       , 2) ,
		('YeastH'                    , 75       , 2) ,   
		('SW-620H'                   , 66       , 2) ,

		( 'amazon0505'               , 96	  , 22),
		( 'artist'                   , 100	  , 12),
		( 'com-amazon'               , 96	  , 22),
		( 'web-BerkStan'             , 100	  , 12),
		( 'soc-BlogCatalog'	         , 128	  , 39),      
		( 'amazon0601'  	         , 96	  , 22), 
		( 'Reddit'                   , 602    , 41),

		# ( 'wiki-topcats'             , 300	  , 12),
		# ( 'COLLAB'                   , 100      , 3) ,
		# ( 'wiki-topcats'             , 300	  , 12),
		# ( 'Reddit'                   , 602      , 41),
		# ( 'enwiki-2013'	           , 100	  , 12),      
		# ( 'amazon_also_bought'       , 96       , 22),
]


data_dir = '/home/yuke/.graphs/orig/'
print(data_dir)
print("dataset,origin,reduced,reduction (%)")


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
	for src_iter in range(0, num_nodes, dense_tile_H):

		dst_list = []
		for src in range(src_iter, src_iter + dense_tile_H):
			dst_list += graph[src]
		range_set = sorted(list(set(dst_list)))
		opt_cnt += (len(range_set) + dense_tile_W - 1)//dense_tile_W
		tmp_opt_cnt = (len(range_set) + dense_tile_W - 1)//dense_tile_W

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

		if tmp < tmp_opt_cnt:
			print(range_set)
			print(tmp, tmp_opt_cnt)
			system.exit(0)
	# tile_cnt = 0
	# for src_iter in range(0, num_nodes, dense_tile_H):
	# 	for dst_iter in range(0, num_nodes, dense_tile_W):
	# 		loc_cnt = 0
	# 		for i in range(src_iter, min(num_nodes, src_iter + dense_tile_H)):
	# 			for j in range(dst_iter, min(num_nodes, dst_iter + dense_tile_W)):
	# 				if i in graph:
	# 					if j in graph[i]:
	# 						tile_cnt += 1
	# 						break

			# tiles.append(loc_cnt)
	print("{}.{},{:},{:.2f}".format(data, tile_cnt, opt_cnt, 100 * (tile_cnt - opt_cnt) / tile_cnt))
						
	# plt.hist(tiles, bins=100)
	# plt.savefig("{}.pdf".format(data))
	# print(Counter(tiles))
	# return tiles

for data, d, c in dataset:
	# print("=> {}".format(data))
	find_dense(data_dir + data, data)