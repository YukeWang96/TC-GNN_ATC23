# sudo ncu --csv -k cusparse --set full python ./train.py --dataset cora --n-hidden 16 --num_classes 6 --n-epochs 1 | tee prof_cora.csv
# sudo ncu --csv -k cusparse --set full python ./train.py --dataset citeseer --n-hidden 16 --num_classes 7 --n-epochs 1 | tee prof_citeseer.csv
# sudo ncu --csv -k cusparse --set full python ./train.py --dataset pubmed --n-hidden 16 --num_classes 3 --n-epochs 1 | tee prof_pubmed.csv

# sudo ncu --csv -k cusparse --set full python ./train.py --dataset citeseer| tee prof_citeseer.csv
# sudo ncu --csv -k cusparse --set full python ./train.py --dataset pubmed| tee  prof_pubmed.csv
# sudo ncu --csv -k cusparse --set full python ./train.py --dataset reddit| tee prof_reddit.csv