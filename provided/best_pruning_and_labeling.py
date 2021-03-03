import numpy as np 

def best_pruning_and_labeling(n,p1,v,T,n_samples):
    """ update admissible A and compute scores; find best pruning and labeling

    :param n: array with number of points sampled in the subtree rooted at each node
    :param p1: array with fraction of label = 1 in the subtree rooted at each node
    :param v: selected subtree Tv's root node
    :param T: tree- 3 element list, see dh.py for description
    :param n_samples: number of samples, 1000

    :returns P_best: the best new pruning
    :returns L_best: the best labeling for v
    """
    p0_tmp = 1 - p1 - ((1/n) + np.sqrt(p1*(1-p1)/n))
    p0_LB = np.fmax(p0_tmp, 0) #fmax is componentwise and ignores nans, unlike max

    p1_tmp = p1 - ((1/n) + np.sqrt(p1*(1-p1)/n))
    p1_LB = np.fmax(p1_tmp, 0)

    A0 = p0_LB > 1/3
    A1 = p1_LB > 1/3
    e1 = 1-p1
    e1_tilde = np.ones(len(e1))
    e1_tilde[A1] = e1[A1]
    e0 = p1
    e0_tilde = np.ones(len(e0))
    e0_tilde[A0] = e1[A0]

    score0 = np.full_like(n,np.nan)
    score1 = np.full_like(n,np.nan)
    score = np.zeros(len(n))

    link = T[0]
    sizes_of_subtrees = T[1] 
    parents = T[2]

    for i in range(len(n)):
        parent = parents[i]
        if parent == 0:
            break
        if A0[i]:
            if np.isnan(score0[parent]):
                score0[parent] = sizes_of_subtrees[i]/sizes_of_subtrees[parent]*score[i]
            else:
                score0[parent] += sizes_of_subtrees[i]/sizes_of_subtrees[parent]*score[i]
        if A1[i]:
            if np.isnan(score1[parent]):
                score1[parent] = sizes_of_subtrees[i]/sizes_of_subtrees[parent]*score[i]
            else:
                score1[parent] += sizes_of_subtrees[i]/sizes_of_subtrees[parent]*score[i]
        possible_scores_tmp = [score0[i],score1[i],e0_tilde[i],e1_tilde[i]]
        score[i] = np.nanmin(possible_scores_tmp)

    possible_scores_tmp = [score0[-1],score1[-1],e0_tilde[-1],e1_tilde[-1]]
    score[-1] = np.nanmin(possible_scores_tmp)

    scores_tmp = [score0[v],score1[v],e0_tilde[v],e1_tilde[v]]
    best = np.nanargmin(scores_tmp)

    if best == 0:
        L_best = 0
        if v < n_samples:
            P_best = np.array([v]) 
        else:
            left = link[v-n_samples,0]
            right = link[v-n_samples,1]
            P_best = np.array([left,right])
    elif best == 1:
        L_best = 1
        if v < n_samples:
            P_best = np.array([v])  
        else:
            left = link[v-n_samples,0]
            right = link[v-n_samples,1]
            P_best = np.array([left,right])
    elif best == 2:
        L_best = 0
        P_best = np.array([v]) 
    else:
        L_best = 1
        P_best = np.array([v]) 

    return P_best, L_best















