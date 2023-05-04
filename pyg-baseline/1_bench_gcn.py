#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

run_GCN = True

dataset = [
        ('citeseer'	        , 3703	    , 6   ),  
        ('cora' 	        , 1433	    , 7   ),  
        ('pubmed'	        , 500	    , 3   ),      
        ('ppi'	            , 50	    , 121 ),   

        ('PROTEINS_full'             , 29       , 2) ,   
        ('OVCAR-8H'                  , 66       , 2) , 
        ('Yeast'                     , 74       , 2) ,
        ('DD'                        , 89       , 2) ,
        ('YeastH'                    , 74       , 2) ,

        ( 'amazon0505'               , 96	, 22),
        ( 'artist'                   , 100      , 12),
        ( 'com-amazon'               , 96	, 22),
        ( 'soc-BlogCatalog'	     , 128      , 39), 
        ( 'amazon0601'  	     , 96	, 22), 
]


for data, d, c in dataset:
    if run_GCN:
        command = "python pyg_main.py --dataset {} \
                --dim {} --classes {} --run_GCN".format(data, d, c)		
    else:
        command = "python pyg_main.py --dataset {} \
                --dim {} --classes {}".format(data, d, c)
    os.system(command)