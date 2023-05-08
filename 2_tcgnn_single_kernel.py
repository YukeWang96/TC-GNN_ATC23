#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

model = 'gcn'
hidden = 16

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

for data, _, _ in dataset:
    command = "python main_tcgnn.py --dataset {} \
									--dim {} \
                                    --hidden {} \
                                    --single_kernel" \
                                    .format(data, hidden, hidden)		
    os.system(command)