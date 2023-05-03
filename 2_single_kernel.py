#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

model = 'gcn'
hidden = 16

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
    command = "python main_tcgnn.py --dataset {} \
                                    --hidden {} \
                                    --single_kernel" \
                                    .format(data, hidden)		
    # command = "sudo ncu --csv --set full " + command 			
    os.system(command)