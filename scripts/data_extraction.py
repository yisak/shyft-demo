import sys
sys.path.insert(0,'D:/users/ysa/shyft_main/shyft')
from os import path
from datetime import datetime

from shyft.orchestration.configuration.yaml_configs import YAMLSimConfig
from shyft.orchestration.simulators.config_simulator import ConfigSimulator

# For testing from statkraft repos (GIS service & SMG)
config_dir = "D:/users/ysa/config/config_test"
config_file = path.join(config_dir, "simulation.yaml")
config_section = "neanidelva"

print('\nConfiguring simulation for region {}'.format(config_section))
cfg = YAMLSimConfig(config_file, config_section)
simulator = ConfigSimulator(cfg)
print('Done initializing...')
simulator.run()
print('Done simulating...')

# 1 - Input climate sources (before interpolation)
precip_src_vct = simulator.region_env.precipitation
nb_precip_src = len(precip_src_vct)
p_idx = 0
p_src = precip_src_vct[p_idx]
p_src_x = p_src.mid_point().x
p_src_y = p_src.mid_point().y
p_src_z = p_src.mid_point().z
p_src_ts = p_src.ts
p_src_v = p_src_ts.values.to_numpy()
p_src_t = [datetime.utcfromtimestamp(p_src_ts.time(i)) for i in range(len(p_src_ts))]

# Cell geo data

# 2 - Interpolated climate inputs

# 2.1 - Aggregated over catchments

# 2.2 - Non-aggregated (at cell level)

# 2.2.1 - Timeseries (for all timsteps) at specific cell

# 2.2.2 - Raster (for all cells) at a specific timestep

# 3 - Responses

# 3.1 - Aggregated over catchments

# 3.2 - Non-aggregated (at cell level)

# 3.2.1 - Timeseries (for all timsteps) at specific cell

# 3.2.2 - Raster (for all cells) at a specific timestep

# 4 - States

# 4.1 - Aggregated over catchments

# 4.2 - Non-aggregated (at cell level)

# 4.2.1 - Timeseries (for all timsteps) at specific cell

# 4.2.2 - Raster (for all cells) at a specific timestep