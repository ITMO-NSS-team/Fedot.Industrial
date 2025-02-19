from typing import Optional

import dask
import pandas as pd
from fedot.core.operations.operation_parameters import OperationParameters
from matplotlib import pyplot as plt

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.transformation.basis.abstract_basis import BasisDecompositionImplementation
from fedot_ind.core.repository.constanst_repository import SPECTRUM_ESTIMATORS


class FourierBasisImplementation(BasisDecompositionImplementation):
    """A class for decomposing data on the Fourier basis and evaluating the derivative of the resulting decomposition.

    Example::
        ts1 = np.random.rand(200)
        ts2 = np.random.rand(200)
        ts = [ts1, ts2]
        bss = FourierBasisImplementation({'threshold': 20000'})
        basis_multi = bss.transform(ts)
        basis_1d = bss.transform(ts1)

    """

    def __repr__(self):
        return 'FourierBasisImplementation'

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.threshold = params.get('threshold', 0.9)
        self.sampling_rate = params.get('sampling_rate', 4096)
        self.output_format = params.get('output_format', 'signal')
        self.approximation = params.get('approximation', 'smooth')
        self.min_rank = params.get('low_rank', 5)

        self.estimator = SPECTRUM_ESTIMATORS[params.get('estimator', 'eigen')]
        self.return_feature_vector = params.get('compute_heuristic_representation', False)
        self.basis = None
        self.filtred_signal = None

        self.logging_params.update({'threshold': self.threshold})

    def _compute_heuristic_features(self, input_data):
        periodogram_class = SPECTRUM_ESTIMATORS['non_parametric']
        estimator = periodogram_class(data=input_data, sampling=self.sampling_rate)
        fft = estimator.psd
        # freq, fft = periodogram(input_data[None, :],
        #                         fs=self.sampling_rate,
        #                         window='hann',
        #                         detrend=False, return_onesided=True, scaling='spectrum', axis=1)
        fft_mean = fft.mean()
        fft_var = fft.var()
        fft_rms = np.sqrt(np.mean(fft ** 2))
        fft_peak_value = fft.max()
        fft_peak_freq = fft[np.argmax(fft)]
        fft_energy = np.sum(fft)
        # features['fft_energy_db'] = 10 * np.log10(fft).sum(axis=1)
        fft_crest_factor = fft_peak_value / fft_rms
        feature_vector = [fft_mean, fft_var, fft_rms, fft_peak_value, fft_peak_freq, fft_energy, fft_crest_factor]
        feature_vector = [round(x, 3) for x in feature_vector]
        return np.array(feature_vector)

    def _visualise_spectrum(self, estimator):
        import matplotlib
        matplotlib.use('TkAgg')
        if isinstance(estimator, np.ndarray):
            pd.DataFrame(estimator).T.plot()
            plt.show()
        else:
            estimator.plot(marker='o')
            plt.show()
        return

    def _build_spectrum(self, input_data):
        estimator = self.estimator(input_data, self.min_rank)
        estimator.run()
        return estimator

    def _decompose_signal(self, input_data):

        estimator = self._build_spectrum(input_data)
        # self._visualise_spectrum(estimator)
        psd = estimator.psd
        if self.return_feature_vector:
            return self._compute_heuristic_features(input_data)
        dominant_freq = np.where(psd >= np.quantile(psd, q=self.threshold))[0]
        if self.approximation == 'exact':
            psd[dominant_freq] = 0
        else:
            psd[~dominant_freq] = 0
        self.filtred_signal = psd if self.output_format == 'spectrum' else np.fft.irfft(psd).reshape(1, -1)
        return self.filtred_signal

    @dask.delayed
    def _transform_one_sample(self, series: np.array):
        return self._get_basis(series)
