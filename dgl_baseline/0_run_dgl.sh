./1_bench_gcn.py| tee 1_bench_gcn.log 2>1_bench_gcn.err
./1_log2csv.py 1_bench_gcn.log

./1_bench_agnn.py| tee 1_bench_agnn.log 2>1_bench_agnn.err
./1_log2csv.py 1_bench_agnn.log

python 2_combine_results.py

mv *.log *.err logs/