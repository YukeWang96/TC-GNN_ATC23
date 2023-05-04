#!/usr/bin/env python3
import os

model = 'agnn'
hidden = [32]
num_layers = 4

dataset = [
		('citeseer'	        , 3703	    , 6   ),  
		('cora' 	        , 1433	    , 7   ),  
		('pubmed'	        , 500	    , 3   ),      
		('ppi'	        	, 50	    , 121 ),   

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


for hid in hidden:
	for data, d, c in dataset:
		print("=> {}, hidden: {}".format(data, hid))
		command = "python train.py \
					--dataset {} \
					--model {}\
					--dim {} \
					--n-hidden {} \
					--n-layers {} \
					--num_classes {}".format(data, model, d, hid, num_layers, c)	
		# command = "ncu --set full -k csrmm_alg2_kernel " + command
		# command = "sudo ncu --csv --set full " + command 				
		os.system(command)
		print()