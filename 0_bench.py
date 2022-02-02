#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

model = 'gcn'
hidden = [16]
num_layers = [2]
single_kernel = True
kernel_profile = False

# model = 'gin'
# hidden = [32]
# num_layers = [4]

dataset = [
        # ('citeseer'	        		, 3703	    , 6   ),  
        # ('cora' 	        		, 1433	    , 7   ),  
        # ('pubmed'	        		, 500	    , 3   ),      
        # ('ppi'	            		, 50	    , 121 ),   
        
        # ('PROTEINS_full'             , 29       , 2) ,   
        # ('OVCAR-8H'                  , 66       , 2) , 
        # ('Yeast'                     , 74       , 2) ,
        # ('DD'                        , 89       , 2) ,
        # ('SW-620H'                   , 66       , 2) ,

        ( 'amazon0505'               , 96	  , 22),
        ( 'artist'                   , 100	  , 12),
        ( 'com-amazon'               , 96	  , 22),
        ( 'soc-BlogCatalog'	         , 128	  , 39),      
        ( 'amazon0601'  	         , 96	  , 22), 

        # ('YeastH'                    , 75       , 2) ,   
        # ( 'web-BerkStan'             , 100	  , 12),
        # ( 'reddit'                   , 602    , 41),
        # ( 'wiki-topcats'             , 300	  , 12),
        # ( 'COLLAB'                   , 100      , 3) ,
        # ( 'wiki-topcats'             , 300	  , 12),
        # ( 'Reddit'                   , 602      , 41),
        # ( 'enwiki-2013'	           , 100	  , 12),      
        # ( 'amazon_also_bought'       , 96       , 22),
]

for n_Layer in num_layers:
    for hid in hidden:
        for data, d, c in dataset:
            print("=> {}, hiddn: {}".format(data, hid))
            command = "python main_tcgnn.py --dataset {} \
                                            --dim {} \
                                            --hidden {} \
                                            --classes {} \
                                            --num_layers {} \
                                            --model {}"\
                    .format(data, hid, hid, c, n_Layer, model)
            if single_kernel:
                command += " --single_kernel"
            if kernel_profile:
                command = "sudo ncu --set detailed --section SpeedOfLight_RooflineChart " + command
            # command = "sudo ncu --csv --set full python main_gcn.py --dataset {0} --dim {1} --hidden {2} --classes {3} --num_layers {4} --model {5} | tee prof_{0}.csv".format(data, d, hid, c, n_Layer, model)		
            os.system(command)