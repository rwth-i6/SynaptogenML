# SynaptogenML

SynaptogenML is a python package based on [Synaptogen](https://github.com/thennen/Synaptogen), which allows to run simulated memristor arrays with PyTorch. It includes code to aid with quantization aware training, as well as memristor-based drop-ins for neural network layers such as the "nn.Linear" and "nn.Conv" modules.

## Usage

SynaptogenML is not a ready-to-use framework, but contains specific modules to allow for a manual modification of existing PyTorch networks. 

## Usage Examples

The `examples` folder contains a toy example in order to understand how SynaptogenML can be used. Please have a look at `create_example_env.sh` to setup a virtual environment for launching the examples.
