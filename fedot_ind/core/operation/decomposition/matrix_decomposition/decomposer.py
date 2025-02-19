from typing import Optional, Union

from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.column_sampling_decomposition import \
    CURDecomposition
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.power_iteration_decomposition import \
    RSVDDecomposition
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.svd_decompostion import SVDDecomposition
from fedot_ind.core.operation.filtration.channel_filtration import _detect_knee_point
from fedot_ind.core.operation.transformation.regularization.spectrum import singular_value_hard_threshold, \
    sv_to_explained_variance_ratio


class MatrixDecomposer:
    """
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        self.decomposition_type = params.get('decomposition_type', 'svd')
        self.decomposition_params = params.get('decomposition_params', 3)
        self.min_components = params.get('min_components_number', None)
        self.spectrum_reg = {'explained_dispersion': sv_to_explained_variance_ratio,
                             'hard_thresholding': singular_value_hard_threshold,
                             'knee_point': _detect_knee_point}
        self.decompose_method = {'svd': SVDDecomposition,
                                 'random_svd': RSVDDecomposition,
                                 'cur': CURDecomposition,
                                 'dmd': None}
        self.decomposition_strategy = self.decompose_method[self.decomposition_type]()

    def spectrum_regularization(self,
                                spectrum: np.array,
                                reg_type: str = 'hard_thresholding'):
        if reg_type in self.spectrum_reg.keys():
            return self.spectrum_reg[reg_type](spectrum)
        else:
            return spectrum

    def get_low_rank(self, spectrum):
        return min(len(spectrum), self.min_components)

    def get_tensor_approximation(self, tensor: Union[dict, np.ndarray]):
        if isinstance(tensor, dict):
            tensor_approx = self.decomposition_strategy.compute_approximation(*tensor.values())
        else:
            tensor_approx = self.decomposition_strategy.compute_approximation(tensor)
        return tensor_approx

    def apply(self, tensor: np.ndarray):
        # Step 1. Get lower bound for rank estimation. By default - 10 % of all data
        self.min_components = int(min(tensor.shape) / 10) if self.min_components is None else self.min_components
        # Step 2. Get a decomposition of original tensor
        U, S, V = self.decomposition_strategy.decompose(tensor)
        # Step 3. Get spectrum regularization. In case of CUR decomposition we dont use it.
        if len(S.shape) != 1:
            stable_rank = self.decomposition_strategy.stable_rank
        else:
            S_reg = self.spectrum_regularization(spectrum=S,
                                                 reg_type=self.decomposition_params['spectrum_regularization'])
            stable_rank = self.get_low_rank(S_reg)
        # Step 4. Get approx data and estimated stable rank.
        result_dict = dict(left_eigenvectors=U,
                           spectrum=S,
                           right_eigenvectors=V,
                           rank=stable_rank)
        # Step 4.1. In case of power iteration with random approx we use computed matrix
        # to rotate original tensor in choosen basis
        if self.decomposition_type.__contains__('random'):
            result_dict['left_eigenvectors'], result_dict['spectrum'], result_dict['right_eigenvectors'] \
                = self.decomposition_strategy.compute_approximation(tensor, result_dict)
        return result_dict
