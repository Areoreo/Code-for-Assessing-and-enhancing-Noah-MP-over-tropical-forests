#!/usr/bin/env python3
"""
Parse Noah-MP NetCDF outputs to CSV format

This script extracts Noah-MP model output from NetCDF format to CSV format
for validation and analysis purposes.

Key Features:
- Preserves original 30-minute temporal resolution by default
- Applies UTC to local time conversion (UTC+8 for Malaysia PSO site)
- Extracts key variables: SOIL_M, LH, HFX, etc.

Timezone Handling:
- Noah-MP outputs timestamps in UTC (UTC+0)
- Malaysia PSO observations are in local time (UTC+8)
- This script converts Noah-MP UTC timestamps to local time by adding 8 hours
  Example: Noah-MP 2005-01-01 04:00 UTC -> 2005-01-01 12:00 local time

Author: Emulator-based calibration project
"""

import sys
import argparse
import xarray as xr
import pandas as pd
import numpy as np
from datetime import timedelta


# =============================================================================
# Timezone Configuration
# =============================================================================
# UTC offset for timestamp conversion
# Set to 0 to keep timestamps in UTC (recommended for consistency)
# The plotting script will handle local time conversion
# Malaysia PSO uses UTC+8, Panama BCI uses UTC-5
UTC_OFFSET_HOURS = 0  # Keep in UTC, let plotting script handle local time conversion


def parse_netcdf_to_csv(nc_file, csv_file, aggregate_daily=False, apply_timezone=True):
    """
    Parse Noah-MP NetCDF output to CSV
    
    Parameters:
    - nc_file: Path to NetCDF file
    - csv_file: Path to output CSV file
    - aggregate_daily: If True, aggregate to daily means (default: False, keep 30-min resolution)
    - apply_timezone: If True, convert UTC to local time (default: True)
    
    Processing Steps:
    1. Open NetCDF file and extract timestamps
    2. Apply timezone conversion (UTC -> local time)
    3. Extract target variables (HFX, LH, SOIL_M, etc.)
    4. Handle multi-dimensional variables (e.g., soil layers)
    5. Optionally aggregate to daily means
    6. Save to CSV
    """
    print(f'Parsing {nc_file}...')
    
    # Step 1: Open NetCDF file with explicit engine
    try:
        ds = xr.open_dataset(nc_file, engine='netcdf4')
    except:
        print('  Trying h5netcdf engine...')
        ds = xr.open_dataset(nc_file, engine='h5netcdf')
    
    # Step 2: Extract and convert timestamps
    # Noah-MP uses 'Times' variable with format 'YYYY-MM-DD_HH:MM:SS'
    times_bytes = ds['Times'].values
    times_str = [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in times_bytes]
    times_utc = pd.to_datetime(times_str, format='%Y-%m-%d_%H:%M:%S')
    
    print(f'  Original time range (UTC): {times_utc.min()} to {times_utc.max()}')
    print(f'  Number of timesteps: {len(times_utc)}')
    
    # Apply timezone conversion: UTC -> local time (UTC+8 for Malaysia PSO)
    if apply_timezone:
        times_local = times_utc + timedelta(hours=UTC_OFFSET_HOURS)
        print(f'  Applied timezone conversion: UTC -> UTC{UTC_OFFSET_HOURS:+d}')
        print(f'  Converted time range (local): {times_local.min()} to {times_local.max()}')
        times = times_local
    else:
        times = times_utc
    
    # Step 3: Create DataFrame with timestamp
    data = {'timestamp': times}
    
    # Step 4: Extract target variables
    # Mapping from Noah-MP variable names to standard names
    var_mapping = {
        'HFX': 'HFX',        # Sensible heat flux (W/m²)
        'LH': 'LH',          # Latent heat flux (W/m²)
        'SOIL_M': 'SOIL_M',  # Soil moisture (m³/m³) - first layer
        'GPP': 'GPP',        # Gross Primary Production (optional)
        'ECAN': 'ECAN',      # Canopy evaporation (mm/s)
        'ETRAN': 'ETRAN',    # Transpiration (mm/s)
        'EDIR': 'EDIR',      # Direct evaporation (mm/s)
        'GRDFLX': 'GRDFLX',  # Ground heat flux (W/m²)
        'FSA': 'FSA',        # Absorbed shortwave radiation (W/m²)
        'FIRA': 'FIRA',      # Net longwave radiation (W/m²)
    }
    
    for noahmp_var, standard_var in var_mapping.items():
        if noahmp_var in ds.variables:
            # Get the variable data
            var_data = ds[noahmp_var].values
            original_shape = var_data.shape
            
            # Squeeze out spatial dimensions (south_north, west_east)
            # For point-scale runs, these are typically size 1
            for _ in range(5):  # Max 5 iterations to prevent infinite loop
                if len(var_data.shape) <= 1:
                    break
                new_shape = var_data.squeeze().shape
                if new_shape == var_data.shape:
                    break
                var_data = var_data.squeeze()
            
            # Handle multi-layer variables (e.g., SOIL_M with soil layers)
            # For SOIL_M, we take only the first layer (top soil)
            if noahmp_var == 'SOIL_M' and len(var_data.shape) > 1:
                # Take first soil layer only
                var_data = var_data[:, 0]
                print(f'  {noahmp_var}: extracted first soil layer, shape {original_shape} -> {var_data.shape}')
            elif len(var_data.shape) > 1:
                # For other multi-dimensional variables, take mean
                for dim_idx in range(len(var_data.shape) - 1, 0, -1):
                    var_data = var_data.mean(axis=dim_idx)
                print(f'  {noahmp_var}: averaged across dimensions, shape {original_shape} -> {var_data.shape}')
            else:
                print(f'  {noahmp_var}: shape {var_data.shape}')
            
            data[standard_var] = var_data
    
    # Step 5: Calculate derived variables
    # Total ET from components (ECAN + ETRAN + EDIR, converted from mm/s to mm/30min)
    if all(k in data for k in ['ECAN', 'ETRAN', 'EDIR']):
        # Convert from mm/s to mm per timestep (30 min = 1800 s)
        data['ET'] = (data['ECAN'] + data['ETRAN'] + data['EDIR']) * 1800
        print(f'  Calculated ET from components (mm/30min)')
    
    # Create DataFrame
    df = pd.DataFrame(data)
    print(f'  Raw data shape: {df.shape}')
    
    # Step 6: Optionally aggregate to daily means
    if aggregate_daily:
        df['date'] = df['timestamp'].dt.date
        
        # Aggregate numeric columns
        numeric_cols = [col for col in df.columns if col not in ['timestamp', 'date']]
        agg_dict = {col: 'mean' for col in numeric_cols}
        
        if agg_dict:
            daily_df = df.groupby('date').agg(agg_dict).reset_index()
            daily_df['timestamp'] = pd.to_datetime(daily_df['date'])
            daily_df = daily_df.drop('date', axis=1)
            
            # Reorder columns
            cols = ['timestamp'] + [col for col in daily_df.columns if col != 'timestamp']
            daily_df = daily_df[cols]
            
            print(f'  Aggregated to daily means: {daily_df.shape}')
            df = daily_df
    
    # Reorder columns to have timestamp first
    cols = ['timestamp'] + [col for col in df.columns if col != 'timestamp']
    df = df[cols]
    
    # Save to CSV
    df.to_csv(csv_file, index=False)
    print(f'  Saved to {csv_file}')
    print(f'  Final columns: {list(df.columns)}')
    print(f'  Time range: {df["timestamp"].min()} to {df["timestamp"].max()}')
    print(f'  Sample data (first 3 rows):')
    print(df.head(3).to_string())
    print()
    
    ds.close()
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Parse Noah-MP NetCDF outputs to CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Default: 30-minute resolution with timezone conversion
  python parse_noahmp_outputs.py --input output.nc --output output.csv
  
  # Aggregate to daily means
  python parse_noahmp_outputs.py --input output.nc --output output.csv --daily
  
  # Keep UTC timestamps (no timezone conversion)
  python parse_noahmp_outputs.py --input output.nc --output output.csv --no-timezone

Timezone:
  Noah-MP outputs use UTC. This script converts to local time (UTC+8 for Malaysia PSO)
  by default. Use --no-timezone to keep original UTC timestamps.
        '''
    )
    parser.add_argument('--input', required=True, help='Input NetCDF file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--daily', action='store_true', 
                       help='Aggregate to daily means (default: keep 30-min resolution)')
    parser.add_argument('--no-timezone', action='store_true',
                       help='Do not apply timezone conversion (keep UTC)')
    
    args = parser.parse_args()
    
    try:
        parse_netcdf_to_csv(
            args.input, 
            args.output, 
            aggregate_daily=args.daily,
            apply_timezone=not args.no_timezone
        )
        print('Success!')
        return 0
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
