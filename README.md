# Welcome to LEnPOD-GP!
LEnPOD-GP is a data-learning based thermal simulator for large-scale many-core chip, derived from [EnPOD-GP](https://www.sciencedirect.com/science/article/pii/S0167926024000658) approach by introducing the concepts of truncation domain and generic truncated domains. LEnPOD-GP approach can be utilized to perform 3D dynamic thermal simulations of  large-scale many-core chips accurately and efficiently. 
This README provides tutorial-like details on how to install and use LEnPOD-GP. It can be further integrated into the performance-power-thermal simulation toolchains as a thermal model. If you use any component of PODTherm-GP, please cite:
```
[1] L. Jiang, A. Dowling, Y. Liu, and Ming-C. Cheng, "Ensemble learning model for effective thermal simulation of multi-core CPUs." Integration 97 (2024): 102201.
[2] L. Jiang, A. Dowling, M. -C. Cheng and Y. Liu, "PODTherm-GP: A Physics-based Data-Driven Approach for Effective Architecture-Level Thermal Simulation of Multi-Core CPUs," in IEEE Transactions on Computers, doi: 10.1109/TC.2023.3278535.
[3] L. Jiang, Y. Liu and M. -C. Cheng, "Fast Accurate Full-Chip Dynamic Thermal Simulation with Fine Resolution Enabled by a Learning Method," in IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, doi: 10.1109/TCAD.2022.3229598.
```
# Dependencies
**FEniCS platform installation**:  
PODTherm-GP thermal simulator is developed on the FEniCS platform(version 2019.1.0), which provides a flexible framework for solving partial differential equations (PDEs) using finite element methods. FEniCS should be pre-installed using the following command:  
```
sudo apt-get install --no-install-recommends software-properties-common  
sudo add-apt-repository ppa:fenics-packages/fenics  
sudo apt-get update  
sudo apt-get install fenics
```
Please refer to the FEniCS installation guide for more detailed instructions on installation and troubleshooting: [FEniCS download](https://fenicsproject.org/download/.).

**Building tools installation**:   
To run the C++ version FEniCS, you need to make sure that the build tools are installed
```
sudo apt install cmake make g++ -y
```
**C++ FEniCS installation**:  
If the cmake are installed on your server, you can then run the following commands to install C++ version FEniCS
```
sudo apt-get install --no-install-recommends software-properties-common
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get install --no-install-recommends fenics
```
# Thermal Length Determination
```
cd Thermal_length
./run.sh
./Cal_length.sh
cd ..

```

# Train IPOD-GP Model
**1. Training Data Collection**
```
cd Training_IPOD-GP/fem_cpp/src
ffc -l dolfin Space.ufl
cd ..  
mkdir build
cd ./build
cmake ..
make 
```
# Construct LEnPOD-GP Model
```
The instruction will be completed soon. A little busy.
```
# Post Processing
