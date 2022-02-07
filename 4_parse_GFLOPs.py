#!/usr/bin/env python3
import sys

if len(sys.argv) < 2:
    raise ValueError("Usage: ./prog [runtime.log]")

fin = open(sys.argv[1])

name_li = []
cusparse_li = []
tcgnn_li = []
for line in fin:
    if "=>" in line:
        name = line.split(',')[0].lstrip("=>")
        name_li.append(name)
    if "dgl.op" in line:
        gflops = line.split(":")[-1].rstrip('\n')
        cusparse_li.append(gflops)
    if "TCGNN" in line:
        gflops = line.split(":")[-1].rstrip('\n')
        tcgnn_li.append(gflops)

fout = open(sys.argv[1]+".csv", "w")   
for name, cusparse, tcgnn in zip(name_li, cusparse_li, tcgnn_li):
    fout.write(name+","+cusparse+","+tcgnn+"\n")
fout.close()