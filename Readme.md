
To run on AWS ubuntu instance. Download the run.sh, activate and run it
```bash
wget https://github.com/pavanreddy-ml/NN-Activation-Function-Computational-Efficiency/raw/main/run.sh
chmod +x run.sh
./run.sh
```

To run on colab. run the following in a cell
```bash
# Mount drive to save results on drive. To save locally remove --colab (Default is False)
from google.colab import drive
drive.mount('/content/drive')

!git clone "https://github.com/pavanreddy-ml/NN-Activation-Function-Computational-Efficiency.git"
!python3 /content/NN-Activation-Function-Computational-Efficiency/Sample_Capstone/code/main.py --colab True
```

Additionally, you can pass in arguments for datasets (Default: ALL), Activations (Default: sigmoid and tanh), batch_size (Default: 1024), colab (True if you want to save results to a drive on colab, else it will save locally on colab and file will be deleted when runtime is deleted), file_num (to number csv as results_<file_num>, to track while distributed testing)
```bash
--datasets "synthetic, mnist_digits, cifar10, california, diabetes, breast_cancer, iris, digits, wine"
--activations "sigmoid, tanh"
--batch_size 1024
--colab Fasle
--file_num 0

```
