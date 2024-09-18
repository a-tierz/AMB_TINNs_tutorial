# TINN_tutorial: A Guide to Thermodynamics-Informed Neural Networks

## Overview
This project implements various neural network models designed to predict and simulate the dynamics of 
simple and double pendulum systems, with and without dissipation, using PyTorch Lightning. 
This tutorial is designed to help you understand and implement Thermodynamics-Informed Neural Networks 
(TINNs), a powerful framework that integrates thermodynamic principles with machine learning models. 
TINNs are particularly useful in cases where enforcing physical laws (such as energy conservation, 
entropy production, and thermodynamic consistency) can enhance the accuracy, interpretability, 
and generalization of neural networks.

This work has been done by:
*   PhD Pau Urdeitx - purdeitx@unizar.es --> [Web](https://amb.unizar.es/people/pau-urdeitx/)
*   PhD Student Alicia Tierz - atierz@unizar.es --> [Web](https://amb.unizar.es/people/alicia-tierz/)
*   PhD Student Carlos Bermejo-Barbanoj - cbbarbanoj@unizar.es --> [Web](https://amb.unizar.es/people/carlos-bermejo-barbanoj/)


The models employed include fully connected feedforward networks (MLPs) and TINNs for both reversible and dissipative systems.

The project is organized to provide:

- Black-box neural network models (e.g., BlackBox) to learn system dynamics from data.
- TINN-based models for learning structured dynamics with physical constraints, particularly for reversible and dissipative systems.
- Solvers for integrating the dynamics over time.

Models Implemented

- BlackBox: A simple feedforward neural network model for learning dynamics in a black-box fashion.
- TINN_01: A TINN model for reversible systems with a fixed skew-symmetric operator.
- TINN_02: A TINN model for reversible systems with a learnable skew-symmetric operator.
- TINN_03: A TINN model that incorporates dissipation using learnable symmetric and skew-symmetric operators.
- TINN: A fully learnable TINN model that learns both reversible and dissipative operators.

## Objectives
The goal is to learn system dynamics for the simple and double pendulum using:

- Energy conservation: For reversible systems, we aim to respect the conservation laws.
- Dissipative dynamics: For systems with dissipation, the models learn to dissipate energy correctly.
- Physical interpretability: Through structured models like TINNs, we attempt to incorporate physically meaningful structures into the neural networks.

## Installation
Prerequisites:
- Python 3.8 or higher
- PyTorch
- PyTorch Lightning
- NumPy
- Matplotlib
- Scikit-learn


You can install the required packages using:

```bash
pip install -r requirements.txt
```


## Key Features
- Custom Neural Architectures: Includes MLP and TINN architectures, optimized for physical systems.
- Reversible and Dissipative Systems: Handles both reversible (energy-conserving) and dissipative dynamics using appropriate operators (L and M matrices).
- Time Integration: Provides solvers to integrate the learned dynamics over time.
- Modularity: The project is highly modular, allowing for easy experimentation with different models and training setups.


## Examples
### BlackBox Model Example

Train the BlackBox model on the double pendulum dataset:

```bash
python main.py --model_name BlackBox --dataset_type no_dissipation --epochs 200 --learning_rate 1e-3
```

### TINN Model with Dissipation
To train the TINN_03 model which includes dissipation:

```bash
python main.py --model_name TINN_03 --dataset_type dissipation --epochs 200 --learning_rate 1e-3
```

## Contributing
Feel free to contribute to this project by submitting a pull request or opening an issue. All contributions are welcome, especially in the areas of model optimization, new solvers, or dataset extensions.

## License
This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). You are free to use, share, and adapt this work for non-commercial purposes, as long as you provide proper attribution. For more details, please refer to the [full license text](https://creativecommons.org/licenses/by-nc/4.0/).

If you need any other sections or further details in the README, let me know, and I’ll be happy to update it!






