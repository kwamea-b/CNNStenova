# CNNStenova
  <img src="CNNStenova_logo.png" width="500" align="right">

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cnnstenova.streamlit.app/) 
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Kernel%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualisation-11557C)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-early%20productionisation-blue)
![Research Software](https://img.shields.io/badge/type-research%20software-purple)
![Educational Tool](https://img.shields.io/badge/use-educational%20tool-green)

CNNStenova is an interactive Streamlit educational and research tool for learning finite-difference stencils and PDE update operators as interpretable convolutional neural-network kernels.
It reuses the analytical solvers, finite-difference solvers, CNN kernel learners and training routines from the [CNN numerical schemes repository](https://github.com/kwamea-b/CNN_numerical_schemes)

## Live demo

CNNStenova is available online here:

https://cnnstenova.streamlit.app/ 

## Features

- Diffusion stencil visualisation
- CNN kernel learning from numerical or analytical PDE data
- Scaled finite-difference stencil recovery
- Von Neumann stability analysis
- Forced Burgers equation CNN kernel learning
- Rollout prediction and error visualisation

## Installation

```bash
pip install -r requirements.txt
