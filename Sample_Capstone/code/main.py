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


import cProfile
from component.activations import *
from component.config import *
from component.nn import *
from component.utils import *
import pstats

results_df = load_file(NEW_RESULTS_FILE)

datasets = ["mnist_digits", "cifar10", "california", "diabetes", "breast_cancer", "iris", "digits", "wine"]
activations = ["tanh"]

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
    elif act == 'tanh':
      c_act = ContinuousTanhNumpy()
    else:
        raise NotImplemented ("Activation not implemented: ", act)
    p_act = ApproximatedNonlinearActivations(lambda x: c_act.forward(x)[0])

    nn_continuous = DenseNN(layer_sizes, activation=c_act.forward, loss=loss)
    nn_piecewise = DenseNN(layer_sizes, activation=p_act.forward, loss=loss)

    def train_network_continuous():
        nn_continuous.train(X, y, learning_rate=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE)

    def train_network_piecewise():
        nn_piecewise.train(X, y, learning_rate=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE)

    run_data = {
    "run": [],
    "continuous_data" : [],
    "continuous_train_time" : [],
    "continuous_loss" : [],
    "piecewise_data" : [],
    "piecewise_train_time" : [],
    "piecewise_loss" : [],
    }

    for i in range(runs_current, RUNS):
      if VERBOSE: print(f"Run: {i+1} of {RUNS} -> Dataset: {dset}, Activation: {act}, Loss: {loss}")
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
              run_data["continuous_data"].append(total_time)

      run_data["continuous_loss"].append(nn_continuous.get_loss(X, y))

      cProfile.run('train_network_piecewise()', 'profile_stats')
      profile_stats = pstats.Stats('profile_stats')
      profile_stats.sort_stats('cumulative')

      for func in profile_stats.fcn_list:
          if func[2] == 'train_network_piecewise':
              func_stats = profile_stats.stats[func]
              cpu_time = func_stats[3]
              run_data["piecewise_train_time"].append(cpu_time)

          if 'forward' == func[2]:
              func_stats = profile_stats.stats[func]
              total_time = func_stats[3]
              run_data["piecewise_data"].append(total_time)

      run_data["piecewise_loss"].append(nn_piecewise.get_loss(X, y))

      run_data["run"].append(i+1)

      if VERBOSE:
        print("Continuous: ", run_data["continuous_data"][-1], "Piecewise: ", run_data["piecewise_data"][-1])

      temp_df = create_dataframe(run_data, dset, act)
      for i in list(run_data.keys()):
        run_data[i] = []
      results_df = concat_results(temp_df, results_df)
      write_results_to_csv(results_df, cache=CAHCE)