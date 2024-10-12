using LearningToOptimize

general_sampler(
    "examples/powermodels/data/6468_rte/6468_rte_SOCWRConicPowerModel_POI_load.mof.json";
    samplers=[
        (original_parameters) -> scaled_distribution_sampler(original_parameters, 10000),
        (original_parameters) -> line_sampler(original_parameters, 1.01:0.01:1.25), 
        (original_parameters) -> box_sampler(original_parameters, 300),
    ],
)