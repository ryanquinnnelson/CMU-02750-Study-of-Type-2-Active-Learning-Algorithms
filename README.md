# CMU-02750-HW2
Spring 2021 - Automation of Scientific Research - course project - Study of Type II Active Learning Algorithms

### Summary
This project implements ZLG and DH algorithms as Python packages with unit tests (pytest) and documentation. The project 
compares performance of active learning algorithms vs offline learning for classification of a sample dataset (nuclear 
vs mitochondrial subcellular localization of protein based on sequence). Two different methods of DH were implemented: 
Case 1 (pruning based on subtree size) and Case 2 (pruning based on selection probability).

Analysis was performed using Jupyter Notebook and Python.


### Project Structure
- DH is implemented as Python package `dh` under /packages.
- ZLG is implemented as Python package `zlg` under /packages.

### Explanation of ZLG

ZLG is a Type II active learning algorithm, used for binary classification on a pool-based data access model. It 
labels instances based on expected risk, and it maintains a prediction for all unlabeled instances in the pool.

The algorithm starts by building a weighted graph over a pool of labeled and unlabeled instances, with edges connecting 
instances which are deemed "similar enough." ZLG uses the graph and known labels to construct a  probability 
distribution over the unlabeled instances. It finds the optimal labeling solution for the distribution in closed-form,
producing label predictions for all unlabeled instances.

ZLG uses these predictions to calculate an expected risk of the labeling. The algorithm searches through the unlabeled 
instances to find the one which reduces the expected risk the most, and selects it for labeling. This process is 
repeated until a defined stopping point, at which point we use the current label predictions to label all unlabeled 
instances.


*In layman's terms, ZLG starts with the idea that if we assume data has natural clusters, we can also assume that 
two data points which are close together are likely to have the same label. Under this theory, labeled instances 
surrounding an unlabeled instance "influence" what the label ought to be, with the amount of influence depending on 
how similar the labeled instance is to the unlabeled one.* 

*ZLG models this influence as a system of energy, with each data point radiating energy outward onto all of its 
neighbors. Data points with similar labels "vibrate" at similar wavelengths, creating resonance; data points 
with different labels vibrate at different wavelengths, creating dissonance. By considering the energies of all 
of the points, we can determine the energy of the overall system.*

*ZLG leverages the well-studied principles of a system of energy (Boltzmann distribution) to efficiently find a 
labeling which produces a system with the least energy (i.e. least dissonance), given a set of labeled instances.*





