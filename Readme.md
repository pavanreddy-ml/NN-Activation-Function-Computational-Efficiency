
To run on AWS ubuntu instance. Download the run.sh, activate and run it
```bash
wget https://github.com/pavanreddy-ml/NN-Activation-Function-Computational-Efficiency/raw/main/run.sh
chmod +x run.sh
./run.sh
```

To run on colab. run the following in a cell
```bash
!git clone "https://github.com/pavanreddy-ml/NN-Activation-Function-Computational-Efficiency.git"
!python3 /content/NN-Activation-Function-Computational-Efficiency/Sample_Capstone/code/main.py
```

Additionally, you can pass in arguments for datasets (Default: ALL), Activations (Default: sigmoid and tanh), batch_size (Default: 1024)
```bash
--datasets "synthetic, mnist_digits, cifar10, california, diabetes, breast_cancer, iris, digits, wine
--activations "sigmoid, tanh"
--batch_size 1024
```
