./0_bench_gcn.py| tee 0_bench_gcn.log 2>0_bench_gcn.err
./1_log2csv.py 0_bench_gcn.log

./0_bench_agnn.py| tee 0_bench_agnn.log 2>0_bench_agnn.err
./1_log2csv.py 0_bench_agnn.log