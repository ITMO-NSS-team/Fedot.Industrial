from py_boost.multioutput.sketching import *

from fedot_ind.core.models.boosting.base_boosting import BaseBoostingModel
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.boosting_rsvd import RandomSVD


class SketchBoostModel(BaseBoostingModel):
    """

    """

    def __init__(self, **kwargs):
        self.industrial_strategy = kwargs['industrial_strategy '] if 'industrial_strategy ' in kwargs.keys() else {}
        if len(self.industrial_strategy) != 0:
            del kwargs['industrial_strategy']
        super().__init__(**kwargs)
        self._init_sketch_method()
        self.params['use_hess'] = self.use_hess
        self.params['multioutput_sketch'] = self.sketch_method

    def _init_sketch_method(self):
        self.sketch_params = self.industrial_strategy.get('sketch_params', {})
        self.sketch_outputs = self.industrial_strategy.get('sketch_outputs', 1)
        self.use_hess = self.industrial_strategy.get('use_hess', False)
        self.sketch_method = self.industrial_strategy.get('sketch_method', 'random_svd')
        self.sketch_method_dict = dict(filter=FilterSketch,
                                       svd=SVDSketch,
                                       random_sampling=RandomSamplingSketch,
                                       random_projection=RandomProjectionSketch,
                                       random_svd=RandomSVD,
                                       )
        self.sketch_method = self.sketch_method_dict[self.sketch_method](self.sketch_outputs,
                                                                         **self.sketch_params)

    def __repr__(self):
        return "SketchBoosting"
