#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

model = 'gcn'
hidden = 256
python_path = "/home/yuke/anaconda3/envs/tcgnn/bin/python main_tcgnn.py"
ncu_path = "/usr/local/cuda-11.3/bin/ncu "
# profiling = True
profiling = False

dataset = [
        # ('toy'	        , 3	    , 2   ),  
        # ('tc_gnn_verify'	, 16	, 2),
        # ('tc_gnn_verify_2x'	, 16	, 2),

        ('citeseer'	        		, 3703	    , 6   ),  
        ('cora' 	        		, 1433	    , 7   ),  
        ('pubmed'	        		, 500	    , 3   ),      
        ('ppi'	            		, 50	    , 121 ),   
        
        # ('PROTEINS_full'             , 29       , 2) ,   
        # ('OVCAR-8H'                  , 66       , 2) , 
        # ('Yeast'                     , 74       , 2) ,
        # ('DD'                        , 89       , 2) ,
        # ('SW-620H'                   , 66       , 2) ,

        # ( 'amazon0505'               , 96	  , 22),
        # ( 'artist'                   , 100	  , 12),
        # ( 'com-amazon'               , 96	  , 22),
        # ( 'soc-BlogCatalog'	         , 128	  , 39),      
        # ( 'amazon0601'  	         , 96	  , 22), 
]

for data, _, _ in dataset:
    command = python_path + " --dataset {} \
                                --dim {} \
                                --single_kernel" \
                                .format(data, hidden)           
    if profiling:
        command = "sudo " + ncu_path + " --set detailed " + command 			
    os.system(command)