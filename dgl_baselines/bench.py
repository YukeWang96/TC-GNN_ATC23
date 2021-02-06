#!/usr/bin/env python3
import subprocess
import datetime
import os

hidden = [16] # [16, 32, 64, 128, 256]  # , 512, 1024, 2048] # [16]
data_dir = '/home/yuke/.graphs/orig/'
print(data_dir)

dataset = [
		# ('toy'	        , 3	    , 2   ),  

		('citeseer'	        , 3703	    , 6   ),  
		# ('cora' 	        , 1433	    , 7   ),  
		# ('pubmed'	        , 500	    , 3   ),      
		# ('ppi'	            , 50	    , 121 ),   

		# ('PROTEINS_full'             , 29       , 2) ,   
		# ('OVCAR-8H'                  , 66       , 2) , 
		# ('Yeast'                     , 74       , 2) ,
		# ('DD'                        , 89       , 2) ,
		# ('YeastH'                    , 75       , 2) ,   
		# ('SW-620H'                   , 66       , 2) ,

		# ( 'amazon0505'               , 96	  , 22),
		# ( 'artist'                   , 100	  , 12),
		# ( 'com-amazon'               , 96	  , 22),
		# ( 'web-BerkStan'             , 100	  , 12),
		# ( 'soc-BlogCatalog'	         , 128	  , 39),      
		# ( 'amazon0601'  	         , 96	  , 22), 
	    # ( 'reddit'                   , 602    , 41),

		# ( 'wiki-topcats'             , 300	  , 12),
		# ( 'COLLAB'                   , 100      , 3) ,
		# ( 'wiki-topcats'             , 300	  , 12),
		# ( 'Reddit'                   , 602      , 41),
		# ( 'enwiki-2013'	           , 100	  , 12),      
		# ( 'amazon_also_bought'       , 96       , 22),
]


for hid in hidden:
	for data, d, c in dataset:
		print("=> {}, hidden: {}".format(data, hid))
		command = "python train.py --dataset {} --dim {} --n-hidden {} --num_classes {}".format(data, d, hid, c)		
		os.system(command)
		print()