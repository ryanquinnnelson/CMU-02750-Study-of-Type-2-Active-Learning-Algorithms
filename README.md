# CMU-02750-HW2
Spring 2021 - Automation of Scientific Research - course project

### Summary
This project implements ZLG and DH algorithms as Python packages with unit tests (pytest) and documentation. The project 
compares performance of active learning algorithms vs offline learning for classification of a sample dataset (nuclear 
vs mitochondrial subcellular localization of protein based on sequence). Two different methods of DH were implemented: 
Case 1 (pruning based on subtree size) and Case 2 (pruning based on selection probability).

Analysis was performed using Jupyter Notebook and Python.


### Project Structure
- DH is implemented as Python package `dh` under /packages.
- ZLG is implemented as Python package `zlg` under /packages.

