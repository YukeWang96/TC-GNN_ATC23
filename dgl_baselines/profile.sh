# sudo ncu --csv -k cusparse --set full python ./train.py --dataset cora| tee prof_cora.csv
# sudo ncu --csv -k cusparse --set full python ./train.py --dataset citeseer| tee prof_citeseer.csv
# sudo ncu --csv -k cusparse --set full python ./train.py --dataset pubmed| tee  prof_pubmed.csv
sudo ncu --csv -k cusparse --set full python ./train.py --dataset reddit| tee prof_reddit.csv