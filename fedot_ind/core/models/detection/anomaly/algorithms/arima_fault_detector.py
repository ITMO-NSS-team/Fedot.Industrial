import pandas as pd
from sympy import Symbol, poly
from typing import Optional, Union, List

from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from sklearn.preprocessing import StandardScaler

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.models.detection.anomaly_detector import AnomalyDetector


class ARIMAFaultDetector(AnomalyDetector):
    """
    ARIMA fault detection algorithm. The idea behind this is to use ARIMA weights
    as features for the anomaly detection algorithm. Using discrete differences of weight coefficients
    for different heuristic methods for obtaining function, which characterized the state using a threshold.

    arimafd source: https://github.com/waico/arimafd
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.ar_order = self.params.get('ar_order', 3)
        self.transformation_mode = 'full'
        self.transformation_type = pd.DataFrame

    def build_model(self):
        return ARIMAAnomalyDetector(ar_order=self.ar_order)

    def predict(self, input_array: InputData) -> np.ndarray:
        converted_input_data = self.convert_input_data(input_array, fit_stage=False)
        prediction = np.zeros(input_array.target.shape)
        labels = self.model_impl.predict(converted_input_data).values.reshape(-1, 1)
        start_idx, end_idx = prediction.shape[0] - labels.shape[0], prediction.shape[0]
        prediction[np.arange(start_idx, end_idx), :] = labels
        return prediction


class ARIMAAnomalyDetector:
    """
    Anomaly detection application of modernized ARIMA model
    """

    def __init__(self, ar_order: Optional[int] = None):
        self.ar_order = ar_order
        self.scaler = StandardScaler()

    def fit(self, history_dataset: Union[pd.Series, pd.DataFrame],
            window: int = 100, window_insensitivity: int = 100) -> pd.Series:
        """
        Fit ARIMA Anomaly detection

        Parameters
        ----------
        history_dataset: pd.Series or pd.DataFrame
            The researched time series or sequences array.
            Desire: dataset without anomalies for computing
            appropriate weights.

        window : int (default=100)
            Time window for calculating metric.
            It will be better if it is equal to
            half the width of the physical process.

        window_insensitivity : int (default=100)
            Аfter the new detected changepoint,
            the following 'window_insensitivity' points
            is guaranteed not to be changepoints.


        Returns
        -------
        bin_metric: pandas array, shape (n_samples), float
            Labeled pandas series, where value 1 is the anomaly,
            0 is not the anomaly.
        """
        self.data = history_dataset
        self.indices = history_dataset.index
        self.generate_tensor(self.ar_order)
        return self.proc_tensor(window=window, window_insensitivity=window_insensitivity)

    def predict(self, data: Union[pd.Series, pd.DataFrame],
                window: int = 100, window_insensitivity: int = 100) -> pd.Series:
        """
        Predicts anomalies by ARIMA

        Parameters
        ----------
        data: pd.Series or pd.DataFrame
            The researched time series or sequences array

        window : int (default=100)
            Time window for calculating metric.
            It will be better if it is equal to
            half the width of the physical process.

        window_insensitivity : int (default=100)
            Аfter the new detected changepoint,
            the following 'window_insensitivity' points
            is guaranteed not to be changepoints.

        Returns
        -------
        bin_metric: pandas array, shape (n_samples), float
            Labeled pandas series, where value 1 is the anomaly,
            0 is not the anomaly.
        """
        self.data = data
        self.indices = data.index

        data = self.scaler.transform(data.copy())

        tensor = np.zeros((data.shape[0],
                           data.shape[1],
                           self.ar_order + 1))
        for i in range(data.shape[1]):
            for value in data[:, i]:
                new_val = self.diffrs[i].transform(value)
                self.models[i].predict(new_val)

            tensor[:, i, :] = self.models[i].dif_w.values[-len(data[:, i]):]
        self.tensor = tensor

        return self.proc_tensor(window=window, window_insensitivity=window_insensitivity)

    def generate_tensor(self, ar_order=None):
        """
        Generation tensor of weights for outlier detection
        """
        data = self.data.copy()

        if ar_order is None:
            ar_order = int(len(data) / 5)
        self.ar_order = ar_order

        data = self.scaler.fit_transform(data.copy())
        tensor = np.zeros((data.shape[0] - ar_order, data.shape[1], ar_order + 1))
        self.models = []
        self.diffrs = []
        for i in range(data.shape[1]):
            diffr = DifferentialIntegrationModule([1])
            dif = diffr.fit_transform(data[:, i])
            self.diffrs.append(diffr)
            model = OnlineTANH(ar_order)
            model.fit(dif)
            self.models.append(model)
            tensor[:, i, :] = model.dif_w.values
        self.tensor = tensor
        return tensor

    def proc_tensor(self, window=100, window_insensitivity=100):
        """
        Processing tensor of weights and calcute metric
        and binary labels

        Parameters
        ----------
        window : int (default=100)
            Time window for calculating metric.
            It will be better if it is equal to
            half the width of the physical process.

        window_insensitivity : int (default=100)
            Аfter the new detected changepoint,
            the following 'window_insensitivity' points
            is guaranteed not to be changepoints.

        Returns
        -------
        bin_metric: pandas array, shape (n_samples), float
            Labeled pandas series, where value 1 is the anomaly,
            0 is not the anomaly.
        """

        tensor = self.tensor.copy()
        df = pd.DataFrame(tensor.reshape(len(tensor), -1), index=self.indices[-len(tensor):])
        metric = (df.rolling(window).max().abs() / df.rolling(window).std().abs()).mean(axis=1)
        ucl = metric.mean() + 3 * metric.std()
        lcl = metric.mean() - 3 * metric.std()
        self.metric = metric
        self.ucl = ucl
        self.lcl = lcl
        bin_metric = ((metric > ucl) | (metric < lcl)).astype(int)
        winn = window_insensitivity
        for i in range(len(bin_metric) - winn):
            if (bin_metric.iloc[i] == 1.0) & (bin_metric[i:i + winn].sum() > 1.0):
                bin_metric[i + 1:i + winn] = np.zeros(winn - 1)
        self.bin_metric = bin_metric
        return bin_metric


class OnlineTANH:
    """
    A class for online arima with stochastic gradient
    descent and log-cosh loss function

    Parameters
    ----------
    order : array-like, shape (default=4)
        Order of autoregression

    lrate : float
        Value of gradient descent rate

    random_state : int, (default=42)
        Random_state is the random number generator

    project, optional (default=True)
        If True, make projection on resolved solution
    """

    def __init__(self, order=4, lrate=0.001, random_state=42, project=True):
        self.order = order
        self.lrate = lrate
        self.random_state = random_state
        self.project = project

    def fit(self, data, init_w=None):
        """
        Fit the AR model according to the given historical data.
        It will be better if data represent normal operation mode

        Parameters
        ----------
        data : array-like, shape (n_samples,)
            Training data, where n_samples is the number of samples

        init_w : array-like, shape (n_weight,), (default=None)
            Initial array of weights, where n_weight is the number of weights
            If None the weights are initialized randomly
        """

        data = np.array(data)
        self.data = data
        np.random.seed(self.random_state)
        self.pred = np.zeros(data.shape[0] + 1) * np.nan
        self.w = np.random.rand(self.order + 1) * 0.01 if init_w is None else init_w.copy()
        self.ww = pd.DataFrame([self.w])
        self.diff = np.zeros(len(self.w))

        self.dif_w = pd.DataFrame([self.w])
        for i in range(self.order, data.shape[0]):
            self.pred[i] = self.w[:-1] @ data[i - self.order:i] + self.w[-1]
            self.diff[:-1] = np.tanh(self.pred[i] - data[i]) * data[i - self.order:i]
            self.diff[-1] = np.tanh(self.pred[i] - data[i])
            self.w -= self.lrate * self.diff

            if self.project:
                self.w = self.projection(self.w)
            self.ww = pd.concat([self.ww, pd.DataFrame([self.w])], ignore_index=True)
            self.dif_w = pd.concat([self.dif_w, pd.DataFrame([self.diff])], ignore_index=True)
        self.pred[-1] = self.w[:-1] @ data[-self.order:] + self.w[-1]

    def predict(self, point_get=None, predict_size=1, return_predict=True):
        """
        Make forecasting series from data to predict_size points

        Parameters
        ----------
        point_get : float (default=None)
            Add new for next iteration of stochastic gradiend descent

        predict_size: float
            The number of out of sample forecasts from the end of the sample

        return_predict: bool (default=True)
            Returns array of forecasting values

        Returns
        -------
        If return_diff = True: data_new : array-like, shape (n_samples - sum_seasons,)
            where sum_seasons is sum of all lags

        """
        if point_get is not None:
            self.data = np.append(self.data, point_get)
            self.diff[:-1] = np.tanh(self.pred[-1] - self.data[-1]) * self.data[-self.order - 1:-1]
            self.diff[-1] = np.tanh(self.pred[-1] - self.data[-1])
            self.w -= self.lrate * self.diff
            if self.project:
                self.w = self.projection(self.w)

            self.ww = pd.concat([self.ww, pd.DataFrame([self.w])], ignore_index=True)

            self.pred = np.append(self.pred, np.nan)
            self.dif_w = pd.concat([self.dif_w, pd.DataFrame([self.diff])], ignore_index=True)
            self.pred[-1] = self.w[:-1] @ self.data[-self.order:] + self.w[-1]

        if predict_size > 1:
            data_p = np.append(self.data[-self.order:], np.zeros(predict_size) * np.nan)

            for i in range(self.order, self.order + predict_size):
                data_p[i] = self.w[:-1] @ data_p[i - self.order:i] + self.w[-1]
            if return_predict:
                return data_p[self.order:]
        elif predict_size == 1 and return_predict:
            return self.pred[-1]

    @staticmethod
    def projection(w, circle=1.01):
        """
        Function for projection weights

        Parameters
        ----------
        w : array-like, shape (n_weights,)
            List of initial weights, where n_weights is the number of weights

        Returns
        -------
        new_w : array-like, shape (n_weights,)
            List of weights resolved solution area,
            where n_weights is the number of weights.
        """
        w = w[::-1]
        roots = np.roots(w)
        l1 = np.linalg.norm(roots)

        if l1 < circle:
            scale = circle / l1
            new_roots = roots * scale
            whole = 1
            x = Symbol('x')
            for root in new_roots:
                whole *= (x - root)
            p = poly(whole, x)
            new_w = np.array([float(coeff.as_real_imag()[0]) for coeff in p.all_coeffs()])
            return new_w[::-1]
        else:
            return w[::-1]


class DifferentialIntegrationModule:
    """
    Differentiation and Integration Module

    This class is needed to bring series to stationarity
    and perform inverse operation.
    """

    def __init__(self, seasons: List[int]):
        self.seasons = seasons

    def fit_transform(self, data, return_diff=True):
        """
        Fit the model and transform data according to the given training data.

        Parameters
        ----------
        data : array-like, shape (n_samples,)
            Training data, where n_samples is the number of samples

        return_diff: optional (default=True)
            Returns the differentiated array

        Returns
        -------
        If return_diff = True: data_new : array-like, shape (n_samples - sum_seasons,)
            where sum_seasons is sum of all lags
        """

        self.data = np.array(data)
        self.minuend = {}
        self.difference = {}
        self.subtrahend = {}
        self.sum_instead_minuend = {}
        self.additional_term = {}

        # process of differentiation
        self.minuend[0] = self.data[self.seasons[0]:]
        self.subtrahend[0] = self.data[:-self.seasons[0]]
        self.difference[0] = self.minuend[0] - self.subtrahend[0]

        self.additional_term[0] = self.data[-self.seasons[0]]
        for i in range(1, len(self.seasons)):
            self.minuend[i] = self.difference[i - 1][self.seasons[i]:]
            self.subtrahend[i] = self.difference[i - 1][:-self.seasons[i]]
            self.difference[i] = self.minuend[i] - self.subtrahend[i]
            self.additional_term[i] = self.difference[i - 1][-self.seasons[i]]

        if return_diff:
            return self.difference[len(self.seasons) - 1]

    def transform(self, point: float):
        """
        Differentiation to the series data that were
        in method fit_transform and plus all the points that
        were in this method.

        Returns
        -------
        Array-like, shape (n_samples + n*n_points - sum_seasons,)
        """
        return self.fit_transform(np.append(self.data, point), return_diff=True)[-1]

    def inverse_fit_transform0(self):
        """
        Return initial data for check class
        """
        self.sum_instead_minuend[len(self.seasons)] = self.difference[len(self.seasons) - 1]
        j = 0
        for i in range(len(self.seasons) - 1, -1, -1):
            self.sum_instead_minuend[i] = self.sum_instead_minuend[i + 1] + self.subtrahend[i][
                sum(self.seasons[::-1][:j]):]
            j += 1
        return self.sum_instead_minuend[0]

    def inverse_transform(self, new_value: float) -> float:
        """
        Return last element after integration.
        (Forecasting value in initial dimension)
       """
        self.new_value = new_value
        self.sum_instead_minuend[len(self.seasons)] = self.new_value
        for i in range(len(self.seasons) - 1, -1, -1):
            self.sum_instead_minuend[i] = self.sum_instead_minuend[i + 1] + self.additional_term[i]

        new_value1 = float(self.sum_instead_minuend[0])
        self.fit_transform(np.append(self.data, new_value1), return_diff=False)
        return new_value1
