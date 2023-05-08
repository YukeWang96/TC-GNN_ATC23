./2_tcgnn_single_kernel.py| tee 2_tcgnn_single_kernel.log 2>2_tcgnn_single_kernel.err
./1_log2csv.py 2_tcgnn_single_kernel.log
mv *.log *.err logs/