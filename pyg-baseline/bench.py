#!/usr/bin/env python3
import datetime
import os
import subprocess

graphs = [
    ('PROTEINS_full'             ),   
    ('OVCAR-8H'                  ),   
    ('Yeast'                     ),   
    ('DD'                        ),
    ('YeastH'                    ),
    ('SW-620H'                   ),

    # ('TWITTER-Real-Graph-Partial'),
    # ('COLLAB'                    ),
    # ('BZR'                       ),
    # ('IMDB-BINARY'               ),
    # ('DD'                        ),

    # ('ENZYMES'                   ),      
    # ('REDDIT-MULTI-12K'          ),       
    # ('REDDIT-MULTI-5K'           ),
    # ('REDDIT-BINARY'             ),
]

for gname in graphs:
    instance = ['python3', 'main.py', '--dataset', gname]
    subprocess.run(instance)