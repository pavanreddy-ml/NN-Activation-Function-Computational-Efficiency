RESULTS_PATH = "results<FILENUM>.csv"
COLAB_RESULTS_PATH = "/content/drive/MyDrive/Research/ANA/results<FILENUM>.csv"
RUNS = 200
VERBOSE = True
OVERWRITE = False
CACHE = False
NEW_RESULTS_FILE = False
EPOCHS = 100
BATCH_SIZE = 1024

COLS = ["dataset", "activation", "run", "continuous_activation_time", "continuous_train_time", "continuous_performance_metric",
        "piecewise_activation_time", "piecewise_train_time", "piecewise_performance_metric", "piecewise_profile_data",
        "combined_activation_time", "combined_train_time", "combined_performance_metric", "combined_profile_data"]
