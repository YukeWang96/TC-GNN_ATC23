#!/usr/bin/env python3
import subprocess
import datetime
import os
from collections import defaultdict
import sys 
import numpy as np
import math

dense_tile_H = 16
dense_tile_W = 8

dataset = [
		('citeseer'	        		, 3703	    , 6   ),  
		('cora' 	        		, 1433	    , 7   ),  
		('pubmed'	        		, 500	    , 3   ),      
		('ppi'	            		, 50	    , 121 ),   
		
		('PROTEINS_full'             , 29       , 2) ,   
		('OVCAR-8H'                  , 66       , 2) , 
		('Yeast'                     , 74       , 2) ,
		('DD'                        , 89       , 2) ,
		('YeastH'                    , 75       , 2) ,   

		( 'amazon0505'               , 96	  , 22),
		( 'artist'                   , 100	  , 12),
		( 'com-amazon'               , 96	  , 22),
		( 'soc-BlogCatalog'	         , 128	  , 39),      
		( 'amazon0601'  	         , 96	  , 22), 
]


data_dir = './tcgnn-ae-graphs/'
print("dataset,origin,reduced,reduction (%)")
fout = open("3_cnt_TC_blk_SpMM.csv", "w")
fout.write("dataset,origin,reduced,reduction (%)\n")

def find_dense(path, data):
	nodes = set()

	graph = defaultdict(list)
	graph_obj = np.load(path+'.npz', allow_pickle=True)
	src_li = graph_obj['src_li']
	dst_li = graph_obj['dst_li']
	num_nodes = graph_obj['num_nodes']

	for src, dst in zip(src_li, dst_li):
		nodes.add(src)
		nodes.add(dst)
		graph[dst].append(src)

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
	
	print("{},{},{},{:.2f}".format(data, tile_cnt, opt_cnt, 100 * (tile_cnt - opt_cnt) / tile_cnt))
	fout = open("3_cnt_TC_blk_SpMM.csv", "a")
	fout.write("{},{},{},{:.2f}\n".format(data, tile_cnt, opt_cnt, 100 * (tile_cnt - opt_cnt) / tile_cnt))

	

if __name__ == '__main__':
	fout = open("3_cnt_TC_blk_SpMM.csv", "w")
	for data, d, c in dataset:
		find_dense(data_dir + data, data)
	fout.close()
	print("\n\nCheck [3_cnt_TC_blk_SpMM.csv] for results.\n\n")