import sys
import argparse
import re
from os import listdir
import json
import xml.etree.ElementTree as ET
import numpy as np
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib
from scipy.stats.mstats import gmean
from statistics import mean

def mean_absolute_percentage_error(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
#  error = np.divide((y_true - y_pred), y_true, out=np.zeros_like(y_true - y_pred), where=y_true!=0)
  error = np.divide((y_true - y_pred), y_true, out=(-y_pred), where=y_true!=0)
  return np.mean(np.abs(error)) * 100
#  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Helper functions to predict certain modules
def BRAM_predict_HLS(dw, depth):
  """ Predict the resource usage of BRAM on Xilinx platforms
  Return the resource usage in a dict

  Args:
    dw: BRAM port width
    depth: BRAM depth
  """
  if dw > 18:
    alpha = np.ceil(dw / 36)
    BRAM = alpha * np.ceil(depth / 512)
  else:
    alpha = np.ceil(dw / 18)
    BRAM = alpha * np.ceil(depth / 1024)
  return BRAM

def URAM_predict_HLS(dw, depth):
  """ Predict the resource usage of URAM on Xilinx platforms
  Return the resource usage in a dict

  Args:
    dw: URAM port width
    depth: URAM depth
  """
  alpha = np.ceil(float(dw) / 72)
  URAM = alpha * np.ceil(float(depth) / 4096)
  return URAM

def BRAM_array_predict_HLS(dw, depth, n_part):
  """ Predict the BRAM resource usage of arrays on Xilinx platform
  Return the resource usage in a dict

  Args:
    dw: BRAM port width (in bytes)
    depth: BRAM depth
    n_part: number of partitions
  """
  return n_part * BRAM_predict_HLS(dw * 8, np.ceil(float(depth) / n_part))

def URAM_array_predict_HLS(dw, depth, n_part):
  return n_part * URAM_predict_HLS(dw * 8, np.ceil(float(depth) / n_part))

#def BRAM_TDP_predict_HLS(dw, depth):
#  """ Predict the resource usage of BRAM in TDP (True dual-port) mode on Xilinx platforms
#  Return the resource usage in a dict
#
#  Args:
#    dw: BRAM port width
#    depth: BRAM depth
#  """
#  s = dw * depth
#  alpha = np.ceil(dw / 18)
#  BRAM = alpha * np.ceil(depth / 1024)
#  return BRAM

def FIFO_predict_xilinx(dw, depth):
  """ Predict the resource ussage of fifo modules on Xilinx platforms
  Return the resource usage in a dict

  Args:
    dw: fifo data width
    depth: fifo depth
  """
  DSP = 0
  if dw * depth <= 512:
    BRAM = 0
    FF = 5
    LUT = dw + 12
  else:
#    BRAM = BRAM_predict_HLS(dw, depth)
    # In the current codegen, we will use SRL to implement FIFOs
    BRAM = 0
    FF = dw + 10
    LUT = int(0.9687 * dw + 13.982)

  return {'BRAM': BRAM, 'DSP': DSP, 'FF': FF, 'LUT': LUT}

def convert_design_infos_to_df(design_infos):
  """ Convert the design info into a dataframe

  The input design_infos is a list.
  Returns a dictionary, containing:
  - df: the dataframe that contains the design inforamtion
  - module_list: all the module names
  - fifo_list: all the fifo names

  Args:
    design_infos: list containing all design infos
  """
  module_list = []
  fifo_list = []
  info_dict = {}
  # Extract module_list and fifo_list
  for design_info in design_infos:
    fifos = design_info['fifos']
    modules = design_info['modules']
    for fifo in fifos:
      if fifo not in fifo_list:
        fifo_list.append(fifo)
    for module in modules:
      if module.find('wrapper') != -1:
        continue

      if module not in module_list:
        module_list.append(module)

  # Reorganize the design infos
  # Initialization
  info_dict['FF'] = []
  info_dict['LUT'] = []
  info_dict['DSP'] = []
  info_dict['BRAM'] = []
  info_dict['URAM'] = []
  # fifos
  for fifo in fifo_list:
    # fifo_cnt, fifo_width, fifo_depth
    info_dict[fifo + '_fifo_cnt'] = []
    info_dict[fifo + '_fifo_width'] = []
    info_dict[fifo + '_fifo_depth'] = []
  # modules
  for module in module_list:
    # IO_module: module_cnt, data_pack_inter, data_pack_intra, ele_type, ele_size
    # [local_buffers_local_X]_{port_width, buffer_depth, partition_number}
    # PE_module: module_cnt, unroll
    if module.find('IO') != -1:
      # IO module
      info_dict[module + '_data_pack_inter'] = []
      info_dict[module + '_data_pack_intra'] = []
      info_dict[module + '_ele_size'] = []
    else:
      # PE module
      info_dict[module + '_unroll'] = []
    info_dict[module + '_module_cnt'] = []
    info_dict[module + '_FF'] = []
    info_dict[module + '_LUT'] = []
    info_dict[module + '_BRAM'] = []
    info_dict[module + '_URAM'] = []
    info_dict[module + '_DSP'] = []

  for design_info in design_infos:
    # FF, LUT, BRAM, DSP
    info_dict['FF'].append(design_info['FF'])
    info_dict['LUT'].append(design_info['LUT'])
    info_dict['DSP'].append(design_info['DSP'])
    info_dict['BRAM'].append(design_info['BRAM'])
    info_dict['URAM'].append(design_info['URAM'])

    fifos = design_info['fifos']
    modules = design_info['modules']
    for fifo in fifo_list:
      if fifo in fifos:
        info_dict[fifo + '_fifo_cnt'].append(fifos[fifo]['fifo_cnt'])
        info_dict[fifo + '_fifo_width'].append(fifos[fifo]['fifo_width'])
        info_dict[fifo + '_fifo_depth'].append(fifos[fifo]['fifo_depth'])
      else:
        info_dict[fifo + '_fifo_cnt'].append(None)
        info_dict[fifo + '_fifo_width'].append(None)
        info_dict[fifo + '_fifo_depth'].append(None)
    for module in module_list:
      if module.find('IO') != -1:
        # IO module
        if module in modules:
          info_dict[module + '_module_cnt'].append(modules[module]['module_cnt'])
          info_dict[module + '_data_pack_inter'].append(modules[module]['data_pack_inter'])
          info_dict[module + '_data_pack_intra'].append(modules[module]['data_pack_intra'])
          info_dict[module + '_ele_size'].append(modules[module]['ele_size'])
        else:
          info_dict[module + '_module_cnt'].append(None)
          info_dict[module + '_data_pack_inter'].append(None)
          info_dict[module + '_data_pack_intra'].append(None)
          info_dict[module + '_ele_size'].append(None)
      else:
        # PE module
        if module in modules:
          info_dict[module + '_module_cnt'].append(modules[module]['module_cnt'])
          info_dict[module + '_unroll'].append(modules[module]['unroll'])
        else:
          info_dict[module + '_module_cnt'].append(None)
          info_dict[module + '_unroll'].append(None)
      if module in modules:
        info_dict[module + '_FF'].append(modules[module]['FF'])
        info_dict[module + '_LUT'].append(modules[module]['LUT'])
        info_dict[module + '_BRAM'].append(modules[module]['BRAM'])
        info_dict[module + '_URAM'].append(modules[module]['URAM'])
        info_dict[module + '_DSP'].append(modules[module]['DSP'])
      else:
        info_dict[module + '_FF'].append(None)
        info_dict[module + '_LUT'].append(None)
        info_dict[module + '_BRAM'].append(None)
        info_dict[module + '_URAM'].append(None)
        info_dict[module + '_DSP'].append(None)

#  print(info_dict['C_IO_L3_in_BRAM'][26])
  # Generate the dataframe
  df = pd.DataFrame(info_dict)
  return {'df': df, 'module_list': module_list, 'fifo_list': fifo_list}

def predict_kernel_DSP_usage(kernel, df, module_list, fifo_list, design_info, verbose=0):
  """ Predict the DSP resource usage for a single design on Xilinx platforms

  Args:
    kernel: kernel_name
    df: dataframe that contains the design information
    module_list: module name list
    fifo_list: fifo name list
    design_info: design information in a dictionary
    verbose: print verbose information
  """
  # Load the resource models
  prj_dir = 'polysa.tmp/optimizer/training/' + kernel + '/resource_models'

#  # predict FIFOs
#  for fifo in fifo_list:
#    if fifo in design_info['fifos']:
#      # Query the library to get the data
#      fifo_w = design_info['fifos'][fifo]['fifo_width'] * 8
#      fifo_depth = design_info['fifos'][fifo]['fifo_depth']
#      resource_info = FIFO_predict_xilinx(fifo_w, fifo_depth)
#      DSP = resource_info['DSP']
#      resource_all[fifo] = {'DSP': DSP, \
#          'n': design_info['fifos'][fifo]['fifo_cnt']}

  module = 'PE'
  if module in design_info['modules']:
    module_feature_set = []
    module_feature_set.append(module + '_unroll')

    # DSP
    X = df.loc[:, module_feature_set]
    model_name = module + '_DSP_model'
    joblib_file = prj_dir + '/' + model_name + '.pkl'
    model = joblib.load(joblib_file)
    DSP = model.predict(X)

    return DSP * design_info['modules'][module]['module_cnt']

#  # predict modules
#  for module in module_list:
#    if module in design_info['modules']:
#      df = feature_extract(df, module)
#      module_feature_set = []
#      if module.find('IO') != -1:
#        module_feature_set.append(module + '_data_pack_inter')
#        module_feature_set.append(module + '_data_pack_inter/' + module + '_data_pack_intra')
#      else:
#        module_feature_set.append(module + '_unroll')
#
#      # DSP
#      X = df.loc[:, module_feature_set]
#      model_name = module + '_DSP_model'
#      joblib_file = prj_dir + '/' + model_name + '.pkl'
#      model = joblib.load(joblib_file)
#      DSP = model.predict(X)
#
#      resource_all[module] = {'DSP': DSP, \
#          'n': design_info['modules'][module]['module_cnt']}
#
#  # Aggregate the resource number
#  # For FF/LUT/DSP prediction, if the module contains inner module, skip it
#  for inst in resource_all:
#    is_outer_module = 0
#    if inst.find('boundary') != -1:
#      if inst[:-9] + '_inter_trans' in resource_all:
#        is_outer_module = 1
#    else:
#      if inst + '_inter_trans' in resource_all:
#        is_outer_module = 1
#    if not is_outer_module:
#      resource['DSP'] += resource_all[inst]['DSP'] * resource_all[inst]['n']

#  return resource['DSP']

def predict_kernel_non_DSP_usage(kernel, df, module_list, fifo_list, design_info, verbose=0):
  """ Predict the resource usage for a single design on Xilinx platforms

  Args:
    kernel: kernel_name
    df: dataframe that contains the design information
    module_list: module name list
    fifo_list: fifo name list
    design_info: design information in a dictionary
    verbose: print verbose information
  """
  # Load the resource models
  prj_dir = 'polysa.tmp/optimizer/training/' + kernel + '/resource_models'
  resource = {'FF': 0, 'LUT': 0, 'BRAM': 0, 'URAM': 0}
  resource_all = {}

  # predict FIFOs
  for fifo in fifo_list:
    if fifo in design_info['fifos']:
      # Query the library to get the data
      fifo_w = design_info['fifos'][fifo]['fifo_width'] * 8
      fifo_depth = design_info['fifos'][fifo]['fifo_depth']
      resource_info = FIFO_predict_xilinx(fifo_w, fifo_depth)
      FF = resource_info['FF']
      LUT = resource_info['LUT']
      BRAM = resource_info['BRAM']
      URAM = 0
      resource_all[fifo] = {'FF': FF, 'LUT': LUT, 'BRAM': BRAM, 'URAM': URAM, \
          'n': design_info['fifos'][fifo]['fifo_cnt']}

  # predict modules
  for module in module_list:
    if module in design_info['modules']:
#      print(module)
#      df.to_csv('debug_predict.csv')
      df = feature_extract(df, module)
      module_feature_set = []
      if module.find('IO') != -1:
        module_feature_set.append(module + '_data_pack_inter')
#        module_feature_set.append(module + '_data_pack_intra')
#        module_feature_set.append(module + '_ele_size')
        module_feature_set.append(module + '_data_pack_inter/' + module + '_data_pack_intra')
#        module_feature_set.append(module + '_data_pack_inter*' + module + '_ele_size')
      else:
        module_feature_set.append(module + '_unroll')

      # FF
      X = df.loc[:, module_feature_set]
      model_name = module + '_FF_model'
      joblib_file = prj_dir + '/' + model_name + '.pkl'
      model = joblib.load(joblib_file)
      FF = model.predict(X)

      # LUT
      X = df.loc[:, module_feature_set]
      model_name = module + '_LUT_model'
      joblib_file = prj_dir + '/' + model_name + '.pkl'
      model = joblib.load(joblib_file)
      LUT = model.predict(X)

      # BRAM
      BRAM = 0
      if 'local_buffers' in design_info['modules'][module]:
        local_buffers = design_info['modules'][module]['local_buffers']
        for local_buffer in local_buffers:
          if local_buffer['mem_type'] == 'BRAM':
            BRAM += BRAM_array_predict_HLS(local_buffer['port_width'], \
                local_buffer['buffer_depth'], local_buffer['partition_number'])
      # URAM
      URAM = 0
      if 'local_buffers' in design_info['modules'][module]:
        local_buffers = design_info['modules'][module]['local_buffers']
        for local_buffer in local_buffers:
          if local_buffer['mem_type'] == 'URAM':
            URAM += URAM_array_predict_HLS(local_buffer['port_width'], \
                local_buffer['buffer_depth'], local_buffer['partition_number'])

      resource_all[module] = {'FF': FF, 'LUT': LUT, 'BRAM': BRAM, 'URAM': URAM, \
          'n': design_info['modules'][module]['module_cnt']}

  # Aggregate the resource number
  # For FF/LUT/DSP prediction, if the module contains inner module, skip it
  for inst in resource_all:
    is_outer_module = 0
    if inst.find('boundary') != -1:
      if inst[:-9] + '_inter_trans' in resource_all:
        is_outer_module = 1
    else:
      if inst + '_inter_trans' in resource_all:
        is_outer_module = 1
    if not is_outer_module:
      resource['FF'] += resource_all[inst]['FF'] * resource_all[inst]['n']
      resource['LUT'] += resource_all[inst]['LUT'] * resource_all[inst]['n']
    resource['BRAM'] += resource_all[inst]['BRAM'] * resource_all[inst]['n']
    resource['URAM'] += resource_all[inst]['URAM'] * resource_all[inst]['n']

  return resource

def predict_kernel_resource_usage(kernel, df, module_list, fifo_list, design_info, verbose=0):
  """ Predict the resource usage for a single design on Xilinx platforms

  Args:
    kernel: kernel_name
    df: dataframe that contains the design information
    module_list: module name list
    fifo_list: fifo name list
    design_info: design information in a dictionary
    verbose: print verbose information
  """
  # Load the resource models
  prj_dir = 'polysa.tmp/optimizer/training/' + kernel + '/resource_models'
  resource = {'FF': 0, 'LUT': 0, 'BRAM': 0, 'URAM': 0, 'DSP': 0}
  resource_all = {}

  # predict FIFOs
  for fifo in fifo_list:
    if fifo in design_info['fifos']:
      # Query the library to get the data
      fifo_w = design_info['fifos'][fifo]['fifo_width'] * 8
      fifo_depth = design_info['fifos'][fifo]['fifo_depth']
      resource_info = FIFO_predict_xilinx(fifo_w, fifo_depth)
      FF = resource_info['FF']
      LUT = resource_info['LUT']
      BRAM = resource_info['BRAM']
      URAM = 0
      DSP = resource_info['DSP']
      resource_all[fifo] = {'FF': FF, 'LUT': LUT, 'BRAM': BRAM, 'URAM': URAM, 'DSP': DSP, \
          'n': design_info['fifos'][fifo]['fifo_cnt']}

  # predict modules
  for module in module_list:
    if module in design_info['modules']:
#      print(module)
#      df.to_csv('debug_predict.csv')
      df = feature_extract(df, module)
      module_feature_set = []
      if module.find('IO') != -1:
        module_feature_set.append(module + '_data_pack_inter')
#        module_feature_set.append(module + '_data_pack_intra')
#        module_feature_set.append(module + '_ele_size')
        module_feature_set.append(module + '_data_pack_inter/' + module + '_data_pack_intra')
#        module_feature_set.append(module + '_data_pack_inter*' + module + '_ele_size')
      else:
        module_feature_set.append(module + '_unroll')

      # FF
      X = df.loc[:, module_feature_set]
      model_name = module + '_FF_model'
      joblib_file = prj_dir + '/' + model_name + '.pkl'
      model = joblib.load(joblib_file)
      FF = model.predict(X)

      # LUT
      X = df.loc[:, module_feature_set]
      model_name = module + '_LUT_model'
      joblib_file = prj_dir + '/' + model_name + '.pkl'
      model = joblib.load(joblib_file)
      LUT = model.predict(X)

      # DSP
      X = df.loc[:, module_feature_set]
      model_name = module + '_DSP_model'
      joblib_file = prj_dir + '/' + model_name + '.pkl'
      model = joblib.load(joblib_file)
      DSP = model.predict(X)

      # BRAM
      BRAM = 0
      if 'local_buffers' in design_info['modules'][module]:
        local_buffers = design_info['modules'][module]['local_buffers']
        for local_buffer in local_buffers:
          if local_buffer['mem_type'] == 'BRAM':
            BRAM += BRAM_array_predict_HLS(local_buffer['port_width'], \
                local_buffer['buffer_depth'], local_buffer['partition_number'])
      # URAM
      URAM = 0
      if 'local_buffers' in design_info['modules'][module]:
        local_buffers = design_info['modules'][module]['local_buffers']
        for local_buffer in local_buffers:
          if local_buffer['mem_type'] == 'URAM':
            URAM += URAM_array_predict_HLS(local_buffer['port_width'], \
                local_buffer['buffer_depth'], local_buffer['partition_number'])

      resource_all[module] = {'FF': FF, 'LUT': LUT, 'BRAM': BRAM, 'URAM': URAM, 'DSP': DSP, \
          'n': design_info['modules'][module]['module_cnt']}

  # Aggregate the resource number
  # For FF/LUT/DSP prediction, if the module contains inner module, skip it
  for inst in resource_all:
    is_outer_module = 0
    if inst.find('boundary') != -1:
      if inst[:-9] + '_inter_trans' in resource_all:
        is_outer_module = 1
    else:
      if inst + '_inter_trans' in resource_all:
        is_outer_module = 1
    if not is_outer_module:
      resource['FF'] += resource_all[inst]['FF'] * resource_all[inst]['n']
      resource['LUT'] += resource_all[inst]['LUT'] * resource_all[inst]['n']
      resource['DSP'] += resource_all[inst]['DSP'] * resource_all[inst]['n']
    resource['BRAM'] += resource_all[inst]['BRAM'] * resource_all[inst]['n']
    resource['URAM'] += resource_all[inst]['URAM'] * resource_all[inst]['n']

  return resource

def feature_extract(df, module):
  if module.find('IO') != -1:
    df[module + '_data_pack_inter/' + module + '_data_pack_intra'] = \
        df.apply(lambda row: float(row[module + '_data_pack_inter']) / float(row[module + '_data_pack_intra']), axis = 1)
    df[module + '_data_pack_inter*' + module + '_ele_size'] = \
        df.apply(lambda row: float(row[module + '_data_pack_inter']) * float(row[module + '_ele_size']), axis = 1)
  return df

def train_module_resource_models(kernel, df, module_list, fifo_list, design_infos, verbose=0):
  """ Train the resource models for each module on Xilinx platforms

  Args:
    kernel: kernel_name
    df: dataframe that contains the design information
    module_list: module name list
    fifo_list: fifo name list
    design_infos: design information in a dictionary
    verbose: print verbose information
  """
  prj_dir = 'polysa.tmp/optimizer/training/' + kernel + '/resource_models'
  # Create the model folder
  sys_cmd = 'rm -rf ' + prj_dir
  print('[PolySA Optimizer] Execute CMD: ' + sys_cmd)
  ret = subprocess.run(sys_cmd.split())

  sys_cmd = 'mkdir ' + prj_dir
  print("[PolySA Optimizer] Execute CMD: " + sys_cmd)
  ret = subprocess.run(sys_cmd.split())

  # Split the train set and validate set
  feature_set = []
  pred_set = []
  for module in module_list:
    df = feature_extract(df, module)
    if module.find('IO') != -1:
      feature_set.append(module + '_data_pack_inter')
#      feature_set.append(module + '_data_pack_intra')
      feature_set.append(module + '_data_pack_inter/' + module + '_data_pack_intra')
#      feature_set.append(module + '_data_pack_inter*' + module + '_ele_size')
#      feature_set.append(module + '_ele_size')
    else:
      feature_set.append(module + '_unroll')
    pred_set.append(module + '_FF')
    pred_set.append(module + '_LUT')
    pred_set.append(module + '_BRAM')
    pred_set.append(module + '_URAM')
    pred_set.append(module + '_DSP')

  X = df.loc[:, feature_set]
  y = df.loc[:, pred_set]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#  print(X_train.head())
#  print(X.head())
#  print(X.iloc[26])

  FF_mape = []
  LUT_mape = []
  DSP_mape = []
  BRAM_mape = []
  URAM_mape = []

##debug
#  print(module_list)

  for module in module_list:
    if verbose:
      print('[PolySA Optimizer] Validate resource model for module: ' + module)
    module_feature_set = []
    if module.find('IO') != -1:
      module_feature_set.append(module + '_data_pack_inter')
#      module_feature_set.append(module + '_data_pack_intra')
      module_feature_set.append(module + '_data_pack_inter/' + module + '_data_pack_intra')
#      module_feature_set.append(module + '_ele_size')
    else:
      module_feature_set.append(module + '_unroll')

    # FF
    module_pred_set = [module + '_FF']
#    print(X_train.head())
    X_train_module = X_train.loc[:, module_feature_set]
    # Remove the missing value
    X_train_module = X_train_module.dropna()
    y_train_module = y_train.loc[:, module_pred_set]
    y_train_module = y_train_module.dropna()
#    print(X_train_module.head())
#    print(y_train_module.head())
    regressor = LinearRegression()
    regressor.fit(X_train_module, y_train_module)
    model = regressor
    model_name = module + '_FF_model'
    joblib_file = prj_dir + '/' + model_name + '.pkl'
    joblib.dump(model, joblib_file)
    # Validate the accuracy
    X_test_module = X_test.loc[:, module_feature_set]
    X_test_module = X_test_module.dropna()
    y_pred_module = model.predict(X_test_module)
    y_test_module = y_test.loc[:, module_pred_set]
    y_test_module = y_test_module.dropna()
    if verbose:
      print('\n======== FF ========\n')
      print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test_module, y_pred_module))
      print('Mean Squared Error: ', metrics.mean_squared_error(y_test_module, y_pred_module))
      print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test_module, y_pred_module)))
      print('Mean Absolute Percentage Error: ', mean_absolute_percentage_error(y_test_module, y_pred_module))
    FF_mape.append(mean_absolute_percentage_error(y_test_module, y_pred_module))

    # LUT
    module_pred_set = [module + '_LUT']
    X_train_module = X_train.loc[:, module_feature_set]
    # Remove the missing value
    X_train_module = X_train_module.dropna()
    y_train_module = y_train.loc[:, module_pred_set]
    y_train_module = y_train_module.dropna()
    regressor = LinearRegression()
    regressor.fit(X_train_module, y_train_module)
    model = regressor
    model_name = module + '_LUT_model'
    joblib_file = prj_dir + '/' + model_name + '.pkl'
    joblib.dump(model, joblib_file)
    # Validate the accuracy
    X_test_module = X_test.loc[:, module_feature_set]
    X_test_module = X_test_module.dropna()
    y_pred_module = model.predict(X_test_module)
    y_test_module = y_test.loc[:, module_pred_set]
    y_test_module = y_test_module.dropna()
    if verbose:
      print('\n======== LUT ========\n')
      print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test_module, y_pred_module))
      print('Mean Squared Error: ', metrics.mean_squared_error(y_test_module, y_pred_module))
      print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test_module, y_pred_module)))
      print('Mean Absolute Percentage Error: ', mean_absolute_percentage_error(y_test_module, y_pred_module))
    LUT_mape.append(mean_absolute_percentage_error(y_test_module, y_pred_module))

    # DSP
    module_pred_set = [module + '_DSP']
    X_train_module = X_train.loc[:, module_feature_set]
    # Remove the missing value
    X_train_module = X_train_module.dropna()
    y_train_module = y_train.loc[:, module_pred_set]
    y_train_module = y_train_module.dropna()
    regressor = LinearRegression()
    regressor.fit(X_train_module, y_train_module)
    model = regressor
    model_name = module + '_DSP_model'
    joblib_file = prj_dir + '/' + model_name + '.pkl'
    joblib.dump(model, joblib_file)
    # Validate the accuracy
    X_test_module = X_test.loc[:, module_feature_set]
    X_test_module = X_test_module.dropna()
    y_pred_module = model.predict(X_test_module)
    y_test_module = y_test.loc[:, module_pred_set]
    y_test_module = y_test_module.dropna()
##debug
#    if module == 'A_IO_L3_in':
#      y_test_module.to_csv('debug_DSP.csv')
    if verbose:
      print('\n======== DSP ========\n')
      print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test_module, y_pred_module))
      print('Mean Squared Error: ', metrics.mean_squared_error(y_test_module, y_pred_module))
      print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test_module, y_pred_module)))
      print('Mean Absolute Percentage Error: ', mean_absolute_percentage_error(y_test_module, y_pred_module))
    DSP_mape.append(mean_absolute_percentage_error(y_test_module, y_pred_module))
##debug
#    if module == 'A_IO_L3_in':
#      print("DSP: ", y_pred_module)

    # BRAM
    module_pred_set = [module + '_BRAM']
    y_test_module = y_test.loc[:, module_pred_set]
    # Validate the accuracy
    X_test_module = X_test_module.dropna()
    y_test_module = y_test_module.dropna()
#    print(y_test_module.head())
##debug
#    if module == 'A_IO_L2_in':
#      y_test_module.to_csv('debug_BRAM.csv')
    y_pred_module = np.zeros((y_test_module.shape[0], 1), dtype = float)
    idx = 0
    for index, row in y_test_module.iterrows():
      design = 'design' + str(index)
#      print(design)
      design_info = design_infos[index]
#      print(design_info)
      BRAM_usage = 0
      if "local_buffers" in design_info['modules'][module]:
        local_buffers = design_info['modules'][module]['local_buffers']
        for local_buffer in local_buffers:
          if local_buffer['mem_type'] == 'BRAM':
            BRAM_usage += BRAM_array_predict_HLS(local_buffer['port_width'], \
                local_buffer['buffer_depth'], local_buffer['partition_number'])
      y_pred_module[idx] = BRAM_usage
      idx += 1
#    if module == 'A_IO_L2_in':
#      print(y_pred_module)

    if verbose:
      print('\n======== BRAM ========\n')
      print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test_module, y_pred_module))
      print('Mean Squared Error: ', metrics.mean_squared_error(y_test_module, y_pred_module))
      print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test_module, y_pred_module)))
      print('Mean Absolute Percentage Error: ', mean_absolute_percentage_error(y_test_module, y_pred_module))
    BRAM_mape.append(mean_absolute_percentage_error(y_test_module, y_pred_module))

    # URAM
    module_pred_set = [module + '_URAM']
    y_test_module = y_test.loc[:, module_pred_set]
    # Validate the accuracy
    X_test_module = X_test_module.dropna()
    y_test_module = y_test_module.dropna()
    y_pred_module = np.zeros((y_test_module.shape[0], 1), dtype = float)
    idx = 0
    for index, row in y_test_module.iterrows():
      design = 'design' + str(index)
      design_info = design_infos[index]
      URAM_usage = 0
      if "local_buffers" in design_info['modules'][module]:
        local_buffers = design_info['modules'][module]['local_buffers']
        for local_buffer in local_buffers:
          if local_buffer['mem_type'] == 'URAM':
            BRAM_usage += URAM_array_predict_HLS(local_buffer['port_width'], \
                local_buffer['buffer_depth'], local_buffer['partition_number'])
      y_pred_module[idx] = URAM_usage
      idx += 1

    if verbose:
      print('\n======== URAM ========\n')
      print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test_module, y_pred_module))
      print('Mean Squared Error: ', metrics.mean_squared_error(y_test_module, y_pred_module))
      print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test_module, y_pred_module)))
      print('Mean Absolute Percentage Error: ', mean_absolute_percentage_error(y_test_module, y_pred_module))
    URAM_mape.append(mean_absolute_percentage_error(y_test_module, y_pred_module))

##debug
#  print(DSP_mape)
#  print(BRAM_mape)

  print('\n======== Module-Level Resource Model Validation Results ========\n')
  print('[PolySA Optimizer] FF Mean Absoulate Percentage Error (Arith. Mean): %.2f%%' %(mean(FF_mape)))
  print('[PolySA Optimizer] LUT Mean Absoulate Percentage Error (Arith. Mean): %.2f%%' %(mean(LUT_mape)))
  print('[PolySA Optimizer] DSP Mean Absoulate Percentage Error (Arith. Mean): %.2f%%' %(mean(DSP_mape)))
  print('[PolySA Optimizer] BRAM Mean Absoulate Percentage Error (Arith. Mean): %.2f%%' %(mean(BRAM_mape)))
  print('[PolySA Optimizer] URAM Mean Absoulate Percentage Error (Arith. Mean): %.2f%%' %(mean(URAM_mape)))

  # Validate on the whole design
  df_test = df.loc[y_test.index.values.tolist(), :]
#  df_test.to_csv('debug.csv')
  FF_design_mape = []
  LUT_design_mape = []
  DSP_design_mape = []
  BRAM_design_mape = []
  URAM_design_mape = []
  for index, row in df_test.iterrows():
#    print(index)
    design_info = design_infos[index]
#    print('row')
#    print(row)
#    print(df_test.head())
    df_design = df_test.loc[[index], :]
#    print(df_design)
    res_usage = predict_kernel_resource_usage(kernel, df_design, module_list, fifo_list, design_info, verbose)
#    print(res_usage)
#debug
    print(index)
    print(res_usage)
    print(design_info['FF'], design_info['LUT'], design_info['BRAM'], design_info['URAM'], design_info['DSP'])
    FF_mape = mean_absolute_percentage_error(np.array([design_info['FF']], dtype='float'), np.array([res_usage['FF']], dtype='float'))
    LUT_mape = mean_absolute_percentage_error(np.array([design_info['LUT']], dtype='float'), np.array([res_usage['LUT']], dtype='float'))
    DSP_mape = mean_absolute_percentage_error(np.array([design_info['DSP']], dtype='float'), np.array([res_usage['DSP']], dtype='float'))
    BRAM_mape = mean_absolute_percentage_error(np.array([design_info['BRAM']], dtype='float'), np.array([res_usage['BRAM']], dtype='float'))
    URAM_mape = mean_absolute_percentage_error(np.array([design_info['URAM']], dtype='float'), np.array([res_usage['URAM']], dtype='float'))
    FF_design_mape.append(FF_mape)
    LUT_design_mape.append(LUT_mape)
    DSP_design_mape.append(DSP_mape)
    BRAM_design_mape.append(BRAM_mape)
    URAM_design_mape.append(URAM_mape)

  print('\n======== Design-Level Resource Model Validation Results ========\n')
  print('[PolySA Optimizer] FF Mean Absoulate Percentage Error (Arith. Mean): %.2f%%' %(mean(FF_design_mape)))
  print('[PolySA Optimizer] LUT Mean Absoulate Percentage Error (Arith. Mean): %.2f%%' %(mean(LUT_design_mape)))
  print('[PolySA Optimizer] DSP Mean Absoulate Percentage Error (Arith. Mean): %.2f%%' %(mean(DSP_design_mape)))
  print('[PolySA Optimizer] BRAM Mean Absoulate Percentage Error (Arith. Mean): %.2f%%' %(mean(BRAM_design_mape)))
  print('[PolySA Optimizer] URAM Mean Absoulate Percentage Error (Arith. Mean): %.2f%%' %(mean(URAM_design_mape)))

def extract_resource_info(hls_rpt):
  """ Extract the resource details from the Xilinx HLS rpt

  Args:
    hls_rpt: root to the hls_rpt in XML format
  """
  resource_info = {'BRAM': 0, 'DSP': 0, 'FF': 0, \
      'LUT': 0, 'URAM': 0}
  root = hls_rpt
  for res_summary in root.iter('AreaEstimates'):
    for child in res_summary:
      if child.tag == 'Resources':
        for item in child:
          if item.tag == 'BRAM_18K':
            resource_info['BRAM'] = int(item.text)
          elif item.tag == 'DSP48E':
            resource_info['DSP'] = int(item.text)
          elif item.tag == 'FF':
            resource_info['FF'] = int(item.text)
          elif item.tag == 'LUT':
            resource_info['LUT'] = int(item.text)
          elif item.tag == 'URAM':
            resource_info['URAM'] = int(item.text)

  return resource_info

def extract_design_info(info_json, info_dat, hls_prj):
  """ Estimate the resource usage of the kernel on Xilinx platform

  Returns a dictionary in the following format:
  - 'FF':
  - 'LUT':
  - 'BRAM':
  - 'DSP':
  - 'fifos': {
    - [fifo_name]: { 'fifo_cnt': x, 'fifo_width': x, 'fifo_depth': x }
    ...
  }
  - 'modules': {
    - [moduel_name]: {
      - 'module_cnt': x,
      - 'FF': x, 'LUT': x, 'BRAM': x, 'DSP': x
      - 'data_pack_inter': x, 'data_pack_intra': x, 'ele_type': x, 'ele_size': x, 'local_buffers': x
      - 'unroll': x, 'local_buffers': x
      }
    ...
  }
  Args:
    info_json: design info in json format
    info_dat: design info in dat format
    hls_prj: directory contains hls project
  """
  # Merge the two design info files
  with open(info_json) as f:
    design_info = json.load(f)
  design_info['fifos'] = {}
  with open(info_dat) as f:
    lines = f.readlines()
  for line in lines:
    line = line.strip().split(':')
    if line[0] == 'fifo':
      # Allocate new entry and insert into the design info
      fifo_name = line[1]
      fifo_cnt = line[2]
      fifo_width = line[3]
      fifo_depth = 2 # default value
      design_info['fifos'][fifo_name] = {}
      design_info['fifos'][fifo_name]['fifo_cnt'] = int(fifo_cnt)
      design_info['fifos'][fifo_name]['fifo_width'] = int(fifo_width)
      design_info['fifos'][fifo_name]['fifo_depth'] = fifo_depth
    elif line[0] == 'module':
      module_name = line[1]
      module_cnt = line[2]
      design_info['modules'][module_name]['module_cnt'] = int(module_cnt)

  # Load the hls project
  hls_rpt_all = {}
  if hls_prj != None:
    hls_rpts = listdir(hls_prj + '/solution1/syn/report')
    hls_rpts = [hls_rpt for hls_rpt in hls_rpts if hls_rpt.endswith('.xml')]
    for f_name in hls_rpts:
      with open(hls_prj + '/solution1/syn/report/' + f_name) as f:
        if f_name.endswith('_csynth.xml'):
          tree = ET.parse(f)
          # Get the root element
          root = tree.getroot()
          # Get the module name
          # Get rid of the '_csynth.xml' suffix
          module_name = f_name[:-11]
          # For duplicate modules, get rid of the digits suffix.
          # Only one report is kept.
          while module_name[-1].isdigit():
            module_name = module_name[:-1]
          hls_rpt_all[module_name] = root

    # Add the resource info into each hw module
    for module_name in design_info['modules']:
      # Grab number from hls report
      if module_name in hls_rpt_all:
        module_hls_rpt = hls_rpt_all[module_name]
      else:
        # It is possible that the module is inlined, look for the wrapper module then
        module_hls_rpt = hls_rpt_all[module_name + '_wrapper']

      resource_info = extract_resource_info(module_hls_rpt)
      FF = resource_info['FF']
      LUT = resource_info['LUT']
      BRAM = resource_info['BRAM']
      URAM = resource_info['URAM']
      DSP = resource_info['DSP']
      design_info['modules'][module_name]['FF'] = FF
      design_info['modules'][module_name]['LUT'] = LUT
      design_info['modules'][module_name]['BRAM'] = BRAM
      design_info['modules'][module_name]['URAM'] = URAM
      design_info['modules'][module_name]['DSP'] = DSP

    # Extract the top kernel
  #  for module in hls_rpt_all:
  #    print(module)
    kernel_hls_rpt = hls_rpt_all['kernel']
    resource_info = extract_resource_info(kernel_hls_rpt)
    design_info['FF'] = resource_info['FF']
    design_info['LUT'] = resource_info['LUT']
    design_info['BRAM'] = resource_info['BRAM']
    design_info['URAM'] = resource_info['URAM']
    design_info['DSP'] = resource_info['DSP']
  else:
    # Add the resource info into each hw module
    for module_name in design_info['modules']:
      # Grab number from hls report
      design_info['modules'][module_name]['FF'] = None
      design_info['modules'][module_name]['LUT'] = None
      design_info['modules'][module_name]['BRAM'] = None
      design_info['modules'][module_name]['URAM'] = None
      design_info['modules'][module_name]['DSP'] = None

    # Extract the top kernel
    design_info['FF'] = None
    design_info['LUT'] = None
    design_info['BRAM'] = None
    design_info['URAM'] = None
    design_info['DSP'] = None

  return design_info

def xilinx_predict(info_json, info_dat, kernel):
  design_info = extract_design_info(info_json, info_dat, None)
  design_infos = [design_info]
  ret = convert_design_infos_to_df(design_infos)
  res_usage = predict_kernel_resource_usage(kernel, ret['df'], ret['module_list'], ret['fifo_list'], design_info, 0)

  return res_usage

def xilinx_run(info_json, info_dat, hls_prj):
  """ Estimate the resource usage of the kernel on Xilinx platform

  Args:
    info_json: design info in json format
    info_dat: design info in dat format
    hls_prj: directory contains hls project
  """
  resource = {"FF": 0, "LUT": 0, "BRAM": 0, "DSP": 0}
  # Merge the two design info files
  with open(info_json) as f:
    design_info = json.load(f)
  design_info['fifos'] = {}
  with open(info_dat) as f:
    lines = f.readlines()
  for line in lines:
    line = line.strip().split(':')
    if line[0] == 'fifo':
      # Allocate new entry and insert into the design info
      fifo_name = line[1]
      fifo_cnt = line[2]
      fifo_width = line[3]
      fifo_depth = 2 # default value
      design_info['fifos'][fifo_name] = {}
      design_info['fifos'][fifo_name]['fifo_cnt'] = int(fifo_cnt)
      design_info['fifos'][fifo_name]['fifo_width'] = int(fifo_width)
      design_info['fifos'][fifo_name]['fifo_depth'] = fifo_depth
    elif line[0] == 'module':
      module_name = line[1]
      module_cnt = line[2]
      design_info['modules'][module_name]['module_cnt'] = int(module_cnt)

  # Load the hls project
  hls_rpts = listdir(hls_prj + '/prj/solution1/syn/report')
  hls_rpts = [hls_rpt for hls_rpt in hls_rpts if hls_rpt.endswith('.xml')]
  hls_rpt_all = {}
  for f_name in hls_rpts:
    with open(hls_prj + '/prj/solution1/syn/report/' + f_name) as f:
      if f_name.endswith('_csynth.xml'):
        tree = ET.parse(f)
        # Get the root element
        root = tree.getroot()
        # Get the module name
        # Get rid of the '_csynth.xml' suffix
        module_name = f_name[:-11]
        # For duplicate modules, get rid of the digits suffix.
        # Only one report is kept.
        while module_name[-1].isdigit():
          module_name = module_name[:-1]
        hls_rpt_all[module_name] = root

  # For each hardware module/fifo in design_info, compute its resource
  resource_all = {}
  fifo_id = 0
  for fifo_name in design_info['fifos']:
    # Query the library to get the data
    fifo_w = design_info['fifos'][fifo_name]['fifo_width'] * 8
    fifo_depth = design_info['fifos'][fifo_name]['fifo_depth']
#    print(str(fifo_w) + ' ' + str(fifo_depth))
    resource_info = FIFO_predict_xilinx(fifo_w, fifo_depth)
    FF = resource_info['FF']
    LUT = resource_info['LUT']
    BRAM = resource_info['BRAM']
    DSP = resource_info['DSP']

    resource_all[fifo_name] = {"FF": FF, "LUT": LUT, "BRAM": BRAM, \
        "DSP": DSP, "n": design_info['fifos'][fifo_name]['fifo_cnt']}
    fifo_id += 1
  module_id = 0
  for module_name in design_info['modules']:
    # Ignore the sub modules
    if module_name.find('inter_trans') != -1 or \
       module_name.find('intra_trans') != -1:
      continue
    # Grab numbers from hls report
    module_hls_rpt = hls_rpt_all[module_name]
    resource_info = extract_resource_info(module_hls_rpt)
    FF = resource_info['FF']
    LUT = resource_info['LUT']
    BRAM = resource_info['BRAM']
    DSP = resource_info['DSP']

    resource_all[module_name] = {"FF": FF, "LUT": LUT, "BRAM": BRAM, \
        "DSP": DSP, "n": design_info['modules'][module_name]['module_cnt']}
    module_id += 1

  # debug
  with open('debug.json', 'w') as f:
    json.dump(resource_all, f, indent=4)

  # Aggregate the resource number
  for inst in resource_all:
    resource['FF'] += resource_all[inst]['FF'] * resource_all[inst]['n']
    resource['LUT'] += resource_all[inst]['LUT'] * resource_all[inst]['n']
    resource['BRAM'] += resource_all[inst]['BRAM'] * resource_all[inst]['n']
    resource['DSP'] += resource_all[inst]['DSP'] * resource_all[inst]['n']

#  # debug
#  with open('debug.json', 'w') as f:
#    json.dump(design_info, f, indent=4)

  return resource

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='==== PolySA Resource Estimator ====')
  parser.add_argument('-j', '--design-info-json', metavar='DESIGN_INFO_JSON', required=True, help='design info json file')
  parser.add_argument('-d', '--design-info-dat', metavar='DESIGN_INFO_DAT', required=True, help='design info dat file')
  parser.add_argument('-pr', '--hls-project', metavar='HLS_PROJECT', required=False, help='directory of HLS project')
  parser.add_argument('-p', '--platform', metavar='PLATFORM', required=True, help='hardware platform: intel/xilinx')
  parser.add_argument('-k', '--kernel', metavar='KERNEL', required=False, default='kernel0', help='kernel name')

  args = parser.parse_args()

  if args.platform == 'intel':
    print('Intel platform not supported yet!')
  elif args.platform == 'xilinx':
#    resource = xilinx_run(args.design_info_json, args.design_info_dat, args.hls_project)
    resource = xilinx_predict(args.design_info_json, args.design_info_dat, args.kernel)
    print("Est. FF: " + str(resource['FF']))
    print("Est. LUT: " + str(resource['LUT']))
    print("Est. BRAM: " + str(resource['BRAM']))
    print("Est. URAM: " + str(resource['URAM']))
    print("Est. DSP: " + str(resource['DSP']))
