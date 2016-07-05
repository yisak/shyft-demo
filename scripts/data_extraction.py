from datetime import datetime
import numpy as np


class DataExtraction(object):
    def __init__(self, simulator):
        self.sim = simulator
        self.r_m = self.sim.region_model
        self.r_env = self.sim.region_env
        self.cells = self.r_m.cells
        self.nb_cells = self.r_m.size()  # len(self.cells)
        self.geo_attr_names = ['x', 'y', 'z', 'area', 'catchment_idx', 'radiation_slope_factor',
                               'glacier_fraction', 'lake_fraction', 'reservoir_fraction', 'forest_fraction',
                               'unknown_fraction']
        self.geo_attr = ['x', 'y', 'z', 'area', 'c_idx', 'rsf', 'gf', 'lf', 'rf', 'ff', 'uf']
        self.c_id_map = self.r_m.catchment_id_map

    def get_input_source(self, src_type):
        src_vct = getattr(self.r_env, src_type)
        src_ts = [src.ts for src in src_vct]
        src_len = [ts.size() for ts in src_ts]
        v = np.array([ts.values.to_numpy() for ts in src_ts])
        t = np.array([[datetime.utcfromtimestamp(ts.time(i)) for i in range(l)] for ts, l in zip(src_ts, src_len)])


# ----------------------------------------------------------------------------------------------------------------------
# 1 - Input climate sources (before interpolation)
# ----------------------------------------------------------------------------------------------------------------------

precip_src_vct = simulator.region_env.precipitation
temp_src_vct = simulator.region_env.temperature
ws_src_vct = simulator.region_env.wind_speed
rh_src_vct = simulator.region_env.rel_hum
rad_src_vct = simulator.region_env.radiation

src_vct = {src_type: getattr(simulator.region_env, src_type)
           for src_type in ['precipitation', 'temperature', 'wind_speed', 'rel_hum', 'radiation']}
precip_src_vct = src_vct['precipitation']

# ----------------------------------------------------------------------------------------------------------------------
# 2 - Cell geo data
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 3 - Interpolated climate inputs
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 3.1 - Aggregated over catchments
# ----------------------------------------------------------------------------------------------------------------------

p_agg_extractor = simulator.region_model.statistics.precipitation
temp_agg_extractor = simulator.region_model.statistics.temperature
rad_agg_extractor = simulator.region_model.statistics.radiation
rh_agg_extractor = simulator.region_model.statistics.rel_hum
ws_agg_extractor = simulator.region_model.statistics.wind_speed

agg_var_extractor = {var_type: getattr(simulator.region_model.statistics, var_type)
           for var_type in ['precipitation', 'temperature', 'wind_speed', 'rel_hum', 'radiation']}
p_agg_extractor = agg_var_extractor['precipitation']

# ----------------------------------------------------------------------------------------------------------------------
# 3.2 - Non-aggregated (at cell level)
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 3.2.1 - Timeseries (for all timsteps) at specific cell
# ----------------------------------------------------------------------------------------------------------------------

def p_cell_extractor(idx): return cells[idx].env_ts.precipitation
def temp_cell_extractor(idx): return cells[idx].env_ts.temperature
def rad_cell_extractor(idx): return cells[idx].env_ts.radiation
def rh_cell_extractor(idx): return cells[idx].env_ts.rel_hum
def ws_cell_extractor(idx): return cells[idx].env_ts.wind_speed

cell_env_ts_extractor = {src_type: lambda idx: getattr(cells[idx].env_ts, src_type)
                         for src_type in ['precipitation', 'temperature', 'wind_speed', 'rel_hum', 'radiation']}
p_cell_extractor = cell_env_ts_extractor['precipitation']

# ----------------------------------------------------------------------------------------------------------------------
# 3.2.2 - Raster (for all cells) at a specific timestep
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# 4 - Responses
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 4.1 - Aggregated over catchments
# ----------------------------------------------------------------------------------------------------------------------

p_agg_extractor = simulator.region_model.statistics.precipitation
temp_agg_extractor = simulator.region_model.statistics.temperature
rad_agg_extractor = simulator.region_model.statistics.radiation
rh_agg_extractor = simulator.region_model.statistics.rel_hum
ws_agg_extractor = simulator.region_model.statistics.wind_speed
q_agg_extractor = simulator.region_model.statistics.discharge

agg_var_extractor = {var_type: getattr(simulator.region_model.statistics, var_type)
           for var_type in ['precipitation', 'temperature', 'wind_speed', 'rel_hum', 'radiation', 'discharge']}
p_agg_extractor = agg_var_extractor['precipitation']
# ----------------------------------------------------------------------------------------------------------------------
# 4.2 - Non-aggregated (at cell level)
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 4.2.1 - Timeseries (for all timsteps) at specific cell
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 4.2.2 - Raster (for all cells) at a specific timestep
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 5 - States
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 5.1 - Aggregated over catchments
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 5.2 - Non-aggregated (at cell level)
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 5.2.1 - Timeseries (for all timsteps) at specific cell
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 5.2.2 - Raster (for all cells) at a specific timestep
# ----------------------------------------------------------------------------------------------------------------------