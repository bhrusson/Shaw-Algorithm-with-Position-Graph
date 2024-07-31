import pickle
str_name = ("./experiments/results/experiment_circuits/"
            "results/toffoli_phir.pkl")
with open(str_name, 'rb') as f:
    data = pickle.load(f)
print(data)