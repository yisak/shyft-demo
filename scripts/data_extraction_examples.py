import sys
sys.path.insert(0,'D:/users/ysa/shyft_main/shyft')
from os import path
from datetime import datetime
import numpy as np

from shyft.orchestration.configuration.yaml_configs import YAMLSimConfig
from shyft.orchestration.simulators.config_simulator import ConfigSimulator

# For testing from statkraft repos (GIS service & SMG)
config_dir = "D:/users/ysa/config/config_test"
config_file = path.join(config_dir, "simulation.yaml")
config_section = "LTM5-Tya"

print('\nConfiguring simulation for region {}'.format(config_section))
cfg = YAMLSimConfig(config_file, config_section)
simulator = ConfigSimulator(cfg)
simulator.region_model.set_state_collection(-1, True)  # enable state collection for all cells
simulator.region_model.set_snow_sca_swe_collection(-1, True)  # enable/disable collection of snow sca|swe for calibration purposes
print('Done initializing...')
simulator.run()
print('Done simulating...')

# ----------------------------------------------------------------------------------------------------------------------
# Getting array with values and timesteps from shyft timeseries extracted in the examples below
# ----------------------------------------------------------------------------------------------------------------------
def get_v_and_t_from_ts(ts):
    return ts.values.to_numpy(), np.array([datetime.utcfromtimestamp(ts.time(i)) for i in range(ts.size())])

# ----------------------------------------------------------------------------------------------------------------------
# Catchment index list, source index, time index and cell index used in the data extraction examples below
# ----------------------------------------------------------------------------------------------------------------------

c_id_select = [177, 172]  # selected catchment IDs
c_id_map = simulator.region_model.catchment_id_map  # all catchmented IDs in region
c_idx_select = np.in1d(c_id_map, c_id_select).nonzero()[0].tolist()  # converting ID to index

idx = 0  # idx-th source (for extraction of a specific input source representing a station or a grid point)
t_idx = 0  # t_idx-th timestep (for extraction of all or selected cell values at a specific timestep)
cell_idx = 0  # cell_tdx-th cell (for extraction of all timesteps at a specific cell)

# ----------------------------------------------------------------------------------------------------------------------
# 1 - Input climate sources (before interpolation)
# ----------------------------------------------------------------------------------------------------------------------

# Get the timeseries for the idx-th precipitation source (station or point)
p_src_ts = simulator.region_env.precipitation[idx].ts

# Get the timeseries for the idx-th temperature source (station or point)
temp_src_ts = simulator.region_env.temperature[idx].ts

# Get the timeseries for the idx-th wind_speed source (station or point)
ws_src_ts = simulator.region_env.wind_speed[idx].ts

# Get the timeseries for the idx-th rel_hum source (station or point)
rh_src_ts = simulator.region_env.rel_hum[idx].ts

# Get the timeseries for the idx-th radiation source (station or point)
rad_src_ts = simulator.region_env.radiation[idx].ts

# Get the x, y and z coordinates for the idx-th precipitation source (station or point)
p_src_geo_pt = simulator.region_env.precipitation[idx].mid_point()
p_src_x, p_src_y, p_src_z = [p_src_geo_pt.x, p_src_geo_pt.y, p_src_geo_pt.z]

# Get the number of precipitation sources
nb_precip_src = len(simulator.region_env.precipitation)

# ----------------------------------------------------------------------------------------------------------------------
# 2 - Cell geo data
# ----------------------------------------------------------------------------------------------------------------------

# Get the geo data and catchment_ID for all cells
#geo_attributes -> ['x', 'y', 'z', 'area', 'catchment_idx', 'radiation_slope_factor',
#                  'glacier_fraction', 'lake_fraction', 'reservoir_fraction', 'forest_fraction', 'unknown_fraction']
cells = simulator.region_model.cells
nb_cells = simulator.region_model.size()  # len(cells)
geo_attr = ['x', 'y', 'z', 'area', 'c_idx', 'rsf', 'gf', 'lf', 'rf', 'ff', 'uf']
nb_geo_attr = len(geo_attr)
cell_geo_data = np.rec.fromrecords(cells.geo_cell_data_vector(cells).to_numpy().reshape(nb_cells, nb_geo_attr),
                                    names=','.join(geo_attr))
catchment_id = c_id_map[cell_geo_data.c_idx.astype(int)]

# ----------------------------------------------------------------------------------------------------------------------
# 3 - Interpolated climate inputs
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 3.1 - Aggregated over catchments
# ----------------------------------------------------------------------------------------------------------------------

# Get the aggregated precipitation timeseries for the whole region (all catchments)
p_agg_all_ts = simulator.region_model.statistics.precipitation([])

# Get the aggregated precipitation timeseries for selected catchments
p_agg_select_ts = simulator.region_model.statistics.precipitation(c_idx_select)

# Get the aggregated temperature timeseries for selected catchments
temp_agg_select_ts = simulator.region_model.statistics.temperature(c_idx_select)

# Get the aggregated rel_hum timeseries for selected catchments
rh_agg_select_ts = simulator.region_model.statistics.rel_hum(c_idx_select)

# Get the aggregated wind_speed timeseries for selected catchments
ws_agg_select_ts = simulator.region_model.statistics.wind_speed(c_idx_select)

# Get the aggregated radiation timeseries for selected catchments
rad_agg_select_ts = simulator.region_model.statistics.radiation(c_idx_select)

# ----------------------------------------------------------------------------------------------------------------------
# 3.2 - Non-aggregated (at cell level)
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 3.2.1 - Timeseries (for all timsteps) at specific cell
# ----------------------------------------------------------------------------------------------------------------------

# Get the precipitation series at the cell_idx-th cell
p_cell_ts = simulator.region_model.cells[cell_idx].env_ts.precipitation

# Get the temperature series at the cell_idx-th cell
temp_cell_ts = simulator.region_model.cells[cell_idx].env_ts.temperature

# Get the wind_speed series at the cell_idx-th cell
ws_cell_ts = simulator.region_model.cells[cell_idx].env_ts.wind_speed

# Get the rel_hum series at the cell_idx-th cell
rh_cell_ts = simulator.region_model.cells[cell_idx].env_ts.rel_hum

# Get the radiation series at the cell_idx-th cell
rad_cell_ts = simulator.region_model.cells[cell_idx].env_ts.radiation

# ----------------------------------------------------------------------------------------------------------------------
# 3.2.2 - Raster (for multiple cells) at a specific timestep
# ----------------------------------------------------------------------------------------------------------------------

# Get the precipitation raster for the whole region (all cells) at the t_idx-th timestep
p_rst_all = simulator.region_model.statistics.precipitation([], t_idx).to_numpy()

# Get the precipitation raster for selected catchments at the t_idx-th timestep
p_rst_select = simulator.region_model.statistics.precipitation(c_idx_select, t_idx).to_numpy()

# Get the temperature raster for selected catchments at the t_idx-th timestep
temp_rst_select = simulator.region_model.statistics.temperature(c_idx_select, t_idx).to_numpy()

# Get the rel_hum raster for selected catchments at the t_idx-th timestep
rh_rst_select = simulator.region_model.statistics.rel_hum(c_idx_select, t_idx).to_numpy()

# Get the wind_speed raster for selected catchments at the t_idx-th timestep
ws_rst_select = simulator.region_model.statistics.wind_speed(c_idx_select, t_idx).to_numpy()

# Get the radiation raster for selected catchments at the t_idx-th timestep
rad_rst_select = simulator.region_model.statistics.radiation(c_idx_select, t_idx).to_numpy()

# ----------------------------------------------------------------------------------------------------------------------
# 4 - Responses
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 4.1 - Aggregated over catchments
# ----------------------------------------------------------------------------------------------------------------------

# Get the aggregated discharge timeseries for the whole region (all catchments)
Qavg_agg_all_ts = simulator.region_model.statistics.discharge([])

# Get the aggregated precipitation timeseries for selected catchments
Qavg_agg_select_ts = simulator.region_model.statistics.discharge(c_idx_select)

# Get the aggregated potential Evapotranspiration timeseries for selected catchments
ETpot_agg_select_ts = simulator.region_model.priestley_taylor_response.output(c_idx_select)

# Get the aggregated actual Evapotranspiration timeseries for selected catchments
ETact_agg_select_ts = simulator.region_model.actual_evaptranspiration_response.output(c_idx_select)

# Get the aggregated snow outflow timeseries for selected catchments
sout_agg_select_ts = simulator.region_model.gamma_snow_response.outflow(c_idx_select)

# Get the aggregated snow water equivalent timeseries for selected catchments
swe_agg_select_ts = simulator.region_model.gamma_snow_response.swe(c_idx_select)

# Get the aggregated snow covered area timeseries for selected catchments
sca_agg_select_ts = simulator.region_model.gamma_snow_response.sca(c_idx_select)

# ----------------------------------------------------------------------------------------------------------------------
# 4.2 - Non-aggregated (at cell level)
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 4.2.1 - Timeseries (for all timsteps) at specific cell
# ----------------------------------------------------------------------------------------------------------------------

# Get the discharge timeseries for the cell_idx-th cell
Qavg_cell_ts = simulator.region_model.cells[cell_idx].rc.avg_discharge

# Get the potential Evapotranspiration timeseries for the cell_idx-th cell
ETpot_cell_ts = simulator.region_model.cells[cell_idx].rc.pe_output

# Get the actual Evapotranspiration timeseries for the cell_idx-th cell
ETact_cell_ts = simulator.region_model.cells[cell_idx].rc.ae_output

# Get the snow outflow timeseries for the cell_idx-th cell
sout_cell_ts = simulator.region_model.cells[cell_idx].rc.snow_outflow

# Get the snow water equivalent timeseries for the cell_idx-th cell
swe_cell_ts = simulator.region_model.cells[cell_idx].rc.snow_swe

# Get the snow covered area timeseries for the cell_idx-th cell
sca_cell_ts = simulator.region_model.cells[cell_idx].rc.snow_sca

# ----------------------------------------------------------------------------------------------------------------------
# 4.2.2 - Raster (for all cells or selected cells) at a specific timestep
# ----------------------------------------------------------------------------------------------------------------------

# Get the discharge raster for the whole region (all catchments)
Qavg_all_rst = simulator.region_model.statistics.discharge([], t_idx).to_numpy()

# Get the discharge raster for selected catchments at the t_idx-th timestep
Qavg_select_rst = simulator.region_model.statistics.discharge(c_idx_select, t_idx).to_numpy()

# Get the potential Evapotranspiration raster for selected catchments at the t_idx-th timestep
ETpot_select_rst = simulator.region_model.priestley_taylor_response.output(c_idx_select, t_idx).to_numpy()

# Get the actual Evapotranspiration raster for selected catchments at the t_idx-th timestep
ETact_select_rst = simulator.region_model.actual_evaptranspiration_response.output(c_idx_select, t_idx).to_numpy()

# Get the snow outflow raster for selected catchments at the t_idx-th timestep
sout_select_rst = simulator.region_model.gamma_snow_response.outflow(c_idx_select, t_idx).to_numpy()

# Get the snow water equivalent raster for selected catchments at the t_idx-th timestep
swe_select_rst = simulator.region_model.gamma_snow_response.swe(c_idx_select, t_idx).to_numpy()

# Get the snow covered area raster for selected catchments at the t_idx-th timestep
sca_select_rst = simulator.region_model.gamma_snow_response.sca(c_idx_select, t_idx).to_numpy()

# ----------------------------------------------------------------------------------------------------------------------
# 5 - States
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 5.1 - Aggregated over catchments
# ----------------------------------------------------------------------------------------------------------------------

# Get the aggregated discharge timeseries for the whole region (all catchments)
Qinst_agg_all_ts = simulator.region_model.kirchner_state.discharge([])

# Get the aggregated precipitation timeseries for selected catchments
Qinst_agg_select_ts = simulator.region_model.kirchner_state.discharge(c_idx_select)

# Get the aggregated acc_melt timeseries for selected catchments
acc_melt_agg_select_ts = simulator.region_model.gamma_snow_state.acc_melt(c_idx_select)

# Get the aggregated albedo timeseries for selected catchments
albedo_agg_select_ts = simulator.region_model.gamma_snow_state.albedo(c_idx_select)

# Get the aggregated alpha timeseries for selected catchments
alpha_agg_select_ts = simulator.region_model.gamma_snow_state.alpha(c_idx_select)

# Get the aggregated iso_pot_energy timeseries for selected catchments
iso_pot_energy_agg_select_ts = simulator.region_model.gamma_snow_state.iso_pot_energy(c_idx_select)

# Get the aggregated lwc timeseries for selected catchments
lwc_agg_select_ts = simulator.region_model.gamma_snow_state.lwc(c_idx_select)

# Get the aggregated surface_heat timeseries for selected catchments
surface_heat_agg_select_ts = simulator.region_model.gamma_snow_state.surface_heat(c_idx_select)

# Get the aggregated temp_swe timeseries for selected catchments
temp_swe_agg_select_ts = simulator.region_model.gamma_snow_state.temp_swe(c_idx_select)

# Get the aggregated sdc_melt_mean timeseries for selected catchments
sdc_melt_mean_agg_select_ts = simulator.region_model.gamma_snow_state.sdc_melt_mean(c_idx_select)

# ----------------------------------------------------------------------------------------------------------------------
# 5.2 - Non-aggregated (at cell level)
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 5.2.1 - Timeseries (for all timsteps) at specific cell
# ----------------------------------------------------------------------------------------------------------------------

# Get the instantaneous discharge timeseries for the cell_idx-th cell
Qinst_cell_ts = simulator.region_model.cells[cell_idx].sc.kirchner_discharge

# Get the acc_melt timeseries for the cell_idx-th cell
acc_melt_cell_ts = simulator.region_model.cells[cell_idx].sc.gs_acc_melt

# Get the albedo timeseries for the cell_idx-th cell
albedo_cell_ts = simulator.region_model.cells[cell_idx].sc.gs_albedo

# Get the alpha timeseries for the cell_idx-th cell
alpha_cell_ts = simulator.region_model.cells[cell_idx].sc.gs_alpha

# Get the iso_pot_energy timeseries for the cell_idx-th cell
iso_pot_energy_cell_ts = simulator.region_model.cells[cell_idx].sc.gs_iso_pot_energy

# Get the lwc timeseries for the cell_idx-th cell
lwc_cell_ts = simulator.region_model.cells[cell_idx].sc.gs_lwc

# Get the sdc_melt_mean timeseries for the cell_idx-th cell
sdc_melt_mean_cell_ts = simulator.region_model.cells[cell_idx].sc.gs_sdc_melt_mean

# Get the surface_heat timeseries for the cell_idx-th cell
surface_heat_cell_ts = simulator.region_model.cells[cell_idx].sc.gs_surface_heat

# Get the temp_swe timeseries for the cell_idx-th cell
temp_swe_cell_ts = simulator.region_model.cells[cell_idx].sc.gs_temp_swe

# ----------------------------------------------------------------------------------------------------------------------
# 5.2.2 - Raster (for all cells or selected cells) at a specific timestep
# ----------------------------------------------------------------------------------------------------------------------

# Get the discharge raster for the whole region (all catchments)
Qinst_all_rst = simulator.region_model.kirchner_state.discharge([], t_idx).to_numpy()

# Get the precipitation raster for selected catchments at the t_idx-th timestep
Qinst_select_rst = simulator.region_model.kirchner_state.discharge(c_idx_select, t_idx).to_numpy()

# Get the acc_melt raster for selected catchments at the t_idx-th timestep
acc_melt_select_rst = simulator.region_model.gamma_snow_state.acc_melt(c_idx_select, t_idx).to_numpy()

# Get the albedo raster for selected catchments at the t_idx-th timestep
albedo_select_rst = simulator.region_model.gamma_snow_state.albedo(c_idx_select, t_idx).to_numpy()

# Get the alpha raster for selected catchments at the t_idx-th timestep
alpha_select_rst = simulator.region_model.gamma_snow_state.alpha(c_idx_select, t_idx).to_numpy()

# Get the iso_pot_energy raster for selected catchments at the t_idx-th timestep
iso_pot_energy_select_rst = simulator.region_model.gamma_snow_state.iso_pot_energy(c_idx_select, t_idx).to_numpy()

# Get the lwc raster for selected catchments at the t_idx-th timestep
lwc_select_rst = simulator.region_model.gamma_snow_state.lwc(c_idx_select, t_idx).to_numpy()

# Get the surface_heat raster for selected catchments at the t_idx-th timestep
surface_heat_select_rst = simulator.region_model.gamma_snow_state.surface_heat(c_idx_select, t_idx).to_numpy()

# Get the temp_swe raster for selected catchments at the t_idx-th timestep
temp_swe_select_rst = simulator.region_model.gamma_snow_state.temp_swe(c_idx_select, t_idx).to_numpy()

# Get the sdc_melt_mean raster for selected catchments at the t_idx-th timestep
sdc_melt_mean_select_rst = simulator.region_model.gamma_snow_state.sdc_melt_mean(c_idx_select, t_idx).to_numpy()