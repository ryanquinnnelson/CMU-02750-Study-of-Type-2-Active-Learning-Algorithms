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
#### Formal Version
ZLG is a Type II active learning algorithm used for binary classification on a pool-based data access model. It 
labels instances based on expected risk, and it comes with no guarantees of performance or label complexity.

The algorithm starts by building a weighted graph over a pool of labeled and unlabeled instances, with edges connecting 
instances which are deemed "similar enough." ZLG uses the graph and known labels to construct a probability 
distribution over the unlabeled instances. It finds the optimal labeling solution for the distribution in closed-form,
producing label predictions for all unlabeled instances.

ZLG uses these predictions to calculate the expected risk of inferring labels for all unlabeled instances.
The algorithm searches through the unlabeled 
instances to find the one which, when labeled, reduces the expected risk the most. ZLG queries the label for this 
instance and updates label
predictions for all unlabeled instances based on the label. This search process is 
repeated until a defined stopping point, at which point we use the current label predictions to label all unlabeled 
instances.

#### Informal Version
*In layman's terms, ZLG starts with the idea that if we assume data has natural clusters, we can also assume that 
two data points which are close together are likely to have the same label. Under this theory, labeled instances 
surrounding an unlabeled instance "influence" what the label ought to be, with the amount of influence depending on 
how similar a labeled instance is to the unlabeled one.* 

*ZLG models this influence as a system of energy, with each data point radiating energy outward onto all of its 
neighbors. Data points with similar labels "vibrate" at similar wavelengths, creating resonance; data points 
with different labels vibrate at different wavelengths, creating dissonance. By considering the energies of all 
of the points, we can determine the energy of the overall system. ZLG leverages the well-studied principles of a 
system of energy (Boltzmann distribution) to efficiently estimate a 
labeling which produces a system with the least energy (i.e. least dissonance), given the current set of labeled 
instances. The estimates represents what the labels should be when considered as a system of energy.*

*Given that this is an estimate, there is some risk that the predicted labels are wrong. Until its query budget is 
used up, ZLG repeatedly searches for the best instances to query to minimize this risk, updating its predictions 
along the way.*


### Explanation of DH
#### Formal Version
DH is a Type II active learning algorithm used for binary and multiclass classification on a pool-based data access 
model. It labels instances based on how likely those instances are to clarify how the data in the pool should be 
clustered in order to minimize error when labels are inferred for unlabeled instances.

DH starts with unlabeled data and performs hierarchical clustering to generate a tree. As data points are grouped 
together, the cluster forms a subtree which contain all of the data points in that group. (The root of the tree forms 
a single subtree which contains all of the data points in the pool.)

The algorithm decides an initial pruning of the tree.

*A pruning is a set of subtrees in the tree which form a disjoint partitioning of all data points. A pruning contains 
one or more subtrees, with each subtree is identified by its root. Each data point belongs to exactly one of these 
subtrees.*

At its stopping point, the algorithm will use the current pruning to set the labels of unlabeled instances. For each
subtree in the pruning, unlabeled instances will be given the majority label of labeled instances in that subtree. DH 
provides a strategy to update the pruning after every query in order to minimize the number of errors during this final 
step.
In order to minimize the 
number of queries made overall, DH also provides a strategy to query the unlabeled points most likely to improve the 
pruning. 

At a high level, the process works as follows. The algorithm takes the current pruning and selects a subtree from it 
based on the size of the subtree vs. the overall tree, as well as the heterogeneity of observed labels in each subtree. (The larger the subtree, the more likely it is 
chosen. The more heterogenous the labels within the subtree, the more likely it is chosen.) An unlabeled point is chosen
at random within the selected subtree for labeling. (This selection strategy corrects for sample bias.) 

Considering this label and all other labeled points in the subtree, DH estimates which action would result in a labeling 
with the least error: (1) retaining this subtree; (2) pruning this subtree into two smaller 
subtrees. Given that subtrees are based on natural clustering, smaller subtrees are more likely to have higher purity.
At the same time, DH wants to minimize the number of subtrees overall: the more subtrees, the more the queries will be 
spread out (and the less DH will know about any given subtree).

After making a decision, the pruning is updated and this process is repeated until the sampling budget has been used. 
Upon reaching the stopping point, DH assigns the majority label to all unlabeled instances in each subtree in the 
pruning. The labeled data set can then be used to train a classifier.


#### Informal Version
*In layman's terms, DH takes a pool of data and figures out the best way to group the data so that the algorithm can 
minimize the number of data points which end up being mislabeled if a single label is assigned to unlabeled
instances in each group.*

*DH starts with all of the data in a single group, and queries the label of some points. Using these labels, DH looks at
the current label mix of the group. If the group is not very pure (i.e. lots of data with different labels), the 
algorithm is incentivized to break the group into two groups using the natural clustering found in the data. (Each 
smaller group is likely 
to have higher purity, which means less error when labels are inferred.) At the same time, DH has an incentive to 
minimize the number of groups overall. The more groups 
DH breaks the data
into, the more the queries will be spread out. (The less queries a group gets, the less certainty there is about the 
purity of the group.) The algorithm considers all factors and decides whether or not to break up the group.*

*Given the resulting group(s), DH is incentivized to select a group which is the less pure and query the label of some
points within that group. DH is also incentivized to select a group which is larger, given that smaller groups tend to 
increase purity overall. This process is repeated until the sample budget is used up. Once this happens, DH takes each of the current groups and assigns
a single label to all unlabeled instances in that group (i.e. the majority label). All data in the 
pool is now labeled, and the dataset can be used for supervised learning.*

