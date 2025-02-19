from benchmark.benchmark_TSER import BenchmarkTSER

init_assumption_pdl = ['quantile_extractor', 'pdl_reg']
init_assumption_rf = ['quantile_extractor', 'treg']
comparasion_dict = dict(pairwise_approach=init_assumption_pdl,
                        baseline=init_assumption_rf)
experiment_setup = {
    'problem': 'regression',
    'metric': 'rmse',
    'timeout': 2.0,
    'num_of_generations': 15,
    'pop_size': 10,
    'metric_names': ('f1', 'accuracy'),
    'logging_level': 10,
    'n_jobs': -1,
    'initial_assumption': comparasion_dict,
    'finetune': True}
custom_dataset = [
    # 'ElectricMotorTemperature',
    #              'PrecipitationAndalusia',
    #  'AcousticContaminationMadrid',
    # 'WindTurbinePower',
    # 'DailyOilGasPrices',
    # 'DailyTemperatureLatitude',
    # 'LPGasMonitoringHomeActivity',
    # 'AluminiumConcentration',
    # 'BoronConcentration',
    # 'CopperConcentration',
    # # 'IronConcentration',
    #  'ManganeseConcentration',
    #  'SodiumConcentration',
    #  'PhosphorusConcentration',
    #  'PotassiumConcentration',
    'MagnesiumConcentration',
    'SulphurConcentration',
    'ZincConcentration',
    'CalciumConcentration'
]
custom_dataset = None
if __name__ == "__main__":
    benchmark = BenchmarkTSER(experiment_setup=experiment_setup,
                              custom_datasets=custom_dataset
                              )
    benchmark.run()
    _ = 1
