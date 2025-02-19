from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.repository.constanst_repository import SINGULAR_VALUE_BETA_THR, SINGULAR_VALUE_MEDIAN_THR


def sv_to_explained_variance_ratio(singular_values):
    """Calculate the explained variance ratio of the singular values.

    Args:
        singular_values (array-like, shape (n_components,)): Singular values.
        rank (int): Number of singular values to use.

    Returns:
        explained_variance (int): Explained variance percent.
        n_components (int): Number of singular values to use.

    """
    singular_values = [abs(x) for x in singular_values]
    n_components = [x / sum(singular_values) * 100 for x in singular_values]
    n_components = [x for x in n_components if x > 3]
    return n_components


def transform_eigen_to_ts(X_elem):
    X_rev = X_elem[::-1]
    eigenvector_to_ts = list(X_rev.diagonal(
        j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1]))
    return eigenvector_to_ts


def eigencorr_matrix(U, S, V,
                     n_components: int = None,
                     correlation_level: float = 0.4):
    d = S.shape[0]
    L = S.shape[0]
    K = V.shape[1]
    if n_components is None:
        n_components = d
    corellated_components = {}
    components_iter = range(n_components)

    X_elem = np.array([S[i] * np.outer(U[:, i], V[i, :]) for i in range(0, d)])

    w = np.array(
        list(np.arange(L) + 1) +  # returns the sequence 1 to L (first line in definition of w)
        [L] * (K - L - 1) +  # repeats L K-L-1 times (second line in w definition)
        list(np.arange(L) + 1)[::-1]  # reverses the first list (equivalent to the third line)
    )

    # Get all the components of the toy series, store them as columns in F_elem array.
    F_elem = np.array([transform_eigen_to_ts(X_elem[i]) for i in range(d)])

    # Calculate the individual weighted norms,
    # ||F_i||_w, first, then take inverse square-root so we don't have to later.
    vector_list = []
    for i in range(d):
        squared_vector = F_elem[i] ** 2
        normed_vector = w.dot(squared_vector)
        vector_list.append(normed_vector)
    F_wnorms = np.array(vector_list)
    F_wnorms = F_wnorms ** -0.5

    # Calculate the w-corr matrix. The diagonal elements are equal to 1, so we can start with an identity matrix
    # and iterate over all pairs of i's and j's (i != j), noting that Wij = Wji.
    Wcorr = np.identity(d)
    for i in range(d):
        for j in range(i + 1, d):
            eigen_vector = F_elem[i]
            next_eigen_vector = F_elem[j]
            Wcorr[i, j] = abs(w.dot(eigen_vector * next_eigen_vector)
                              * F_wnorms[i] * F_wnorms[j])
            Wcorr[j, i] = Wcorr[i, j]

    component_set = [x for x in components_iter]
    for i in components_iter:
        component_idx = np.where(Wcorr[i] > correlation_level)[0]
        intersect = set(component_set).intersection(component_idx)
        have_intersection = len(intersect) != 0
        if have_intersection:
            for j in component_idx.tolist():
                if j in component_set:
                    component_set.remove(j)
            corellated_components.update({f'{i}_component': component_idx})
        else:
            continue

    return corellated_components


def singular_value_hard_threshold(singular_values,
                                  rank=None,
                                  beta=None,
                                  threshold=SINGULAR_VALUE_MEDIAN_THR) -> list:
    """Calculate the hard threshold for the singular values.

    Args:
        singular_values (array-like, shape (n_components,)): Singular values.
        rank (int): Number of singular values to use.
        beta (float): Beta value.
        threshold (float): Threshold value.

    Returns:
        adjusted singular values array (array-like, shape (n_components,)): Adjusted array of singular values.

    """
    if rank is not None:
        return singular_values[:rank]
    else:
        # Find the median of the singular values
        singular_values = [s_val for s_val in singular_values if s_val > 0.01]
        if len(singular_values) == 1:
            return singular_values[:1]
        median_sv = np.median(singular_values[:rank])
        # Find the adjusted rank
        if threshold is None:
            threshold = SINGULAR_VALUE_BETA_THR(beta)
        sv_threshold = threshold * median_sv
        # Find the threshold value
        adjusted_rank = np.sum(singular_values >= sv_threshold)
        # If the adjusted rank is 0, recalculate the threshold value
        if adjusted_rank == 0:
            sv_threshold = 2.31 * median_sv
            adjusted_rank = max(np.sum(singular_values >= sv_threshold), 1)
        return singular_values[:adjusted_rank]


def reconstruct_basis(U, Sigma, VT, ts_length):
    # check whether Sigma value is set to 'ill_conditioned'
    if isinstance(Sigma, str):
        # rank = round(len(VT)*0.1)
        rank = len(VT)
        TS_comps = np.zeros((ts_length, rank))
        U, S, V = U[0], U[1], U[2]
        for idx, (comp, eigen_idx) in enumerate(VT.items()):
            X_dominant = np.sum([S[i] * np.outer(U[:, i], V[i, :]) for i in eigen_idx], axis=0)
            grouped_eigenvector = transform_eigen_to_ts(X_dominant)
            if idx == rank:
                break
            else:
                TS_comps[:, idx] = grouped_eigenvector
        TS_comps[:, 1] = np.sum(TS_comps[:, 1:], axis=1)
        TS_comps = TS_comps[:, :2]
        return TS_comps
    if len(Sigma.shape) > 1:
        def multi_reconstruction(x):
            return reconstruct_basis(U=U, Sigma=x, VT=VT, ts_length=ts_length)

        TS_comps = list(map(multi_reconstruction, Sigma))
    else:
        rank = Sigma.shape[0]
        TS_comps = np.zeros((ts_length, rank))
        for i in range(rank):
            X_elem = Sigma[i] * np.outer(U[:, i], VT[i, :])
            X_rev = X_elem[::-1]
            eigenvector = [X_rev.diagonal(
                j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]
            TS_comps[:, i] = eigenvector
    return TS_comps
