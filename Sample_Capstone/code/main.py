import os
NUM_THREADS = '1'
os.environ['MKL_NUM_THREADS'] = NUM_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = NUM_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = NUM_THREADS
os.environ['OMP_NUM_THREADS'] = NUM_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = NUM_THREADS

import psutil
p = psutil.Process(os.getpid())
p.cpu_affinity([0])

import json
import argparse
import cProfile
from activations import *
from config import *
from nn import *
from utils import *
import pstats
import numpy as np

results_df = load_file(NEW_RESULTS_FILE)

if __name__ == "__main__":
    datasets = ["synthetic", "mnist_digits", "cifar10", "california", "diabetes", "breast_cancer", "iris", "digits",
                "wine"]
    activations = ["sigmoid", "tanh"]

    parser = argparse.ArgumentParser(description='Process input arguments')

    parser.add_argument('--datasets', type=str, help='Dataset to use', required=False)
    parser.add_argument('--activations', type=str, help='Activation function to use', required=False)
    parser.add_argument('--batch_size', type=int, help='Batch size for training', required=False, default=BATCH_SIZE)
    parser.add_argument('--colab', type=bool, help='Whether colab notebook or not', required=False, default=False)
    parser.add_argument('--file_num', type=int, help='file num for parallel num. Default 0', required=False, default=0)

    args = parser.parse_args()

    if args.colab == True:
        from google.colab import drive
        drive.mount('/content/drive')
        RESULTS_PATH = COLAB_RESULTS_PATH

    if args.file_num == 0:
        RESULTS_PATH.replace("<FILENUM>", "")
    else:
        RESULTS_PATH.replace("<FILENUM>", str(args.file_num))

    if args.datasets is not None:
        ds = args.datasets.split(",")
        ds = [i.strip() for i in ds]
        invalid = []
        for i in ds:
            if i not in datasets:
                invalid.append(i)
        if len(invalid) != 0:
            raise ValueError("Invalid Dataset(s): " + ', '.join(invalid))
        datasets = ds

    if args.activations is not None:
        a = args.activations.split(",")
        a = [i.strip() for i in a]
        invalid = []
        for i in a:
            if i not in activations:
                invalid.append(i)
        if len(invalid) != 0:
            raise ValueError("Invalid Activation(s): " + ', '.join(invalid))
        activations = a

    BATCH_SIZE = args.batch_size

    for act in activations:
        for dset in datasets:
            runs_current = 0
            if OVERWRITE:
                results_df = results_df[~((results_df["dataset"] == dset) & (results_df["activation"] == act))]
            else:
                if len(results_df[(results_df["dataset"] == dset) & (results_df["activation"] == act)]) == RUNS:
                    continue
                else:
                    runs_current = len(results_df[(results_df["dataset"] == dset) & (results_df["activation"] == act)])

            X, y, layer_sizes, loss = preprocess_data(dset)

            if act == 'sigmoid':
                c_act = ContinuousSigmoidNumpy()
                p_act = ApproximatedNonlinearActivations(lambda x: 1 / (1 + np.exp(-x)))
                comb_act = ApproximatedGradientActivations(lambda x: 1 / (1 + np.exp(-x)))
            elif act == 'tanh':
                c_act = ContinuousTanhNumpy()
                p_act = ApproximatedNonlinearActivations(lambda x: np.tanh(x))
                comb_act = ApproximatedGradientActivations(lambda x: np.tanh(x))
            else:
                raise NotImplemented("Activation not implemented: ", act)


            nn_continuous = DenseNN(layer_sizes, activation=c_act.forward, loss=loss)
            nn_piecewise = DenseNN(layer_sizes, activation=p_act.forward, loss=loss)
            nn_combined = DenseNN(layer_sizes, activation=p_act.forward, loss=loss)


            def train_network_continuous():
                nn_continuous.train(X, y, learning_rate=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE)

            def train_network_piecewise():
                nn_piecewise.train(X, y, learning_rate=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE)

            def train_network_combined():
                nn_combined.train(X, y, learning_rate=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE)


            run_data = {
                "run": [],
                "continuous_activation_time": [],
                "continuous_train_time": [],
                "continuous_loss": [],
                "piecewise_activation_time": [],
                "piecewise_train_time": [],
                "piecewise_loss": [],
                "piecewise_profile_data": [],
                "combined_activation_time": [],
                "combined_train_time": [],
                "combined_loss": [],
                "combined_profile_data": []
            }

            for i in range(runs_current, RUNS):
                if VERBOSE: print(f"Run: {i + 1} of {RUNS} -> Dataset: {dset}, Activation: {act}, Loss: {loss}")
                cProfile.run('train_network_continuous()', 'profile_stats')
                profile_stats = pstats.Stats('profile_stats')
                profile_stats.sort_stats('cumulative')

                for func in profile_stats.fcn_list:
                    if func[2] == 'train_network_continuous':
                        func_stats = profile_stats.stats[func]
                        cpu_time = func_stats[3]
                        run_data["continuous_train_time"].append(cpu_time)

                    if 'forward' == func[2]:
                        func_stats = profile_stats.stats[func]
                        total_time = func_stats[3]
                        run_data["continuous_activation_time"].append(total_time)

                run_data["continuous_loss"].append(nn_continuous.get_loss(X, y))

                cProfile.run('train_network_piecewise()', 'profile_stats')
                profile_stats = pstats.Stats('profile_stats')
                profile_stats.sort_stats('cumulative')


                prof_data = {}
                for func in profile_stats.fcn_list:
                    if func[2] in ["<method 'astype' of 'numpy.ndarray' objects>", "clip", "<method 'take' of 'numpy.ndarray' objects>", "forward", "train_network_piecewise"]:
                        if func[2] == "<method 'astype' of 'numpy.ndarray' objects>":
                            prof_data["astype"] = profile_stats.stats[func][3]
                        if func[2] == "clip":
                            prof_data["clip"] = profile_stats.stats[func][3]
                        if func[2] == "<method 'take' of 'numpy.ndarray' objects>":
                            prof_data["indexing"] = profile_stats.stats[func][3]
                        if func[2] == "forward":
                            func_stats = profile_stats.stats[func]
                            total_time = func_stats[3]
                            run_data["piecewise_activation_time"].append(total_time)
                        if func[2] == "train_network_piecewise":
                            func_stats = profile_stats.stats[func]
                            cpu_time = func_stats[3]
                            run_data["piecewise_train_time"].append(cpu_time)

                prof_data["arithmetic"] = run_data["piecewise_activation_time"][-1] - (sum(prof_data.values()))
                run_data["piecewise_profile_data"] = json.dumps(prof_data)
                run_data["piecewise_loss"].append(nn_piecewise.get_loss(X, y))

                cProfile.run('train_network_combined()', 'profile_stats')
                profile_stats = pstats.Stats('profile_stats')
                profile_stats.sort_stats('cumulative')

                prof_data = {}
                for func in profile_stats.fcn_list:
                    if func[2] in ["<method 'astype' of 'numpy.ndarray' objects>", "clip",
                                   "<method 'take' of 'numpy.ndarray' objects>", "forward", "train_network_combined"]:
                        if func[2] == "<method 'astype' of 'numpy.ndarray' objects>":
                            prof_data["astype"] = profile_stats.stats[func][3]
                        if func[2] == "clip":
                            prof_data["clip"] = profile_stats.stats[func][3]
                        if func[2] == "<method 'take' of 'numpy.ndarray' objects>":
                            prof_data["indexing"] = profile_stats.stats[func][3]
                        if func[2] == "forward":
                            func_stats = profile_stats.stats[func]
                            total_time = func_stats[3]
                            run_data["combined_activation_time"].append(total_time)
                        if func[2] == "train_network_combined":
                            func_stats = profile_stats.stats[func]
                            cpu_time = func_stats[3]
                            run_data["combined_train_time"].append(cpu_time)

                prof_data["arithmetic"] = run_data["combined_activation_time"][-1] - (sum(prof_data.values()))
                run_data["combined_profile_data"] = json.dumps(prof_data)
                run_data["combined_loss"].append(nn_piecewise.get_loss(X, y))
                run_data["run"].append(i + 1)

                if VERBOSE:
                    print("Continuous: ", run_data["continuous_activation_time"][-1], "Piecewise: ",
                          run_data["piecewise_activation_time"][-1], "Combined: ", run_data["combined_activation_time"][-1])

                temp_df = create_dataframe(run_data, dset, act)
                for i in list(run_data.keys()):
                    run_data[i] = []
                results_df = concat_results(temp_df, results_df)
                write_results_to_csv(results_df, cache=CACHE)
