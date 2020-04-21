import sys
import argparse
import re
import os
import json
import subprocess
import itertools
import numpy as np
import resource_est as res_model
import latency_est as latency_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib
import xml.etree.ElementTree as ET
import time
import multiprocessing
import random
from statistics import mean
import copy

def mean_absolute_percentage_error(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def generate_sa_sizes_cmd(sa_sizes):
  """ Generate the command line argument to specify the sa_sizes
  Concatenate each size in the sa_sizes to generate the final argument.

  Args:
    sa_sizes: containing the sizes for each optimization stage
  """
  length = len(sa_sizes)
  first = 1
  cmd = '--sa-sizes='
#  if length > 1:
#    cmd += '\"'
  cmd += '{'
  for size in sa_sizes:
    if not first:
      cmd += ';'
    cmd += size
    first = 0

  cmd += '}'
#  if length > 1:
#    cmd += '\"'
  return cmd

def generate_loop_candidates(loops, config, left, right, loop_limit=-1):
  """ Generate candidate loops given the loop bounds
  This function samples each loop dimension given the sample numbers set in the config,
  then builds the cartesian product to generate all possible loop tuples to traverse.
  Example: given the input loops as [4, 4], and n_sample as 2 set in config, it will first sample
  each loop dimension and generate two lists [2, 4], [2, 4]. Then, it computes the cartesian product
  and generates all candidates: [[2, 2], [2, 4], [4, 2], [4, 4]].

  Due to the current implementation limitation, we have the following limitation on the loop candidates:
  - For array partitioninig, the loop candidates should be left-exclusive and right-inclusive.
    This prevents generating single PEs along certain direction which causes codegen breakdown.
  - For latency hiding, the loop candidates should be left-inclusive and right-exclusive.
    Similarly, making it right-exclusive to prevent possible single PE situation.
  - For SIMD, the loop candidates will be both left-inclusive and right-inclusive.
  - For both latency hiding and SIMD, if we choose tiling factor as 1, the corresponding opt. step will be
    automatically skipped in PolySA. The reason is that for latency hiding, tiling factor as 1 couldn't help.
    And for SIMD, the computation is inherently with SIMD lane as 1.

  If config['setting']['sample_mode'] is "log", we will generate samples as exponentials of 2, i.e. [1, 2, 4, 8].
  Otherwise, if config['setting']['sample_mode'] is "linear", we will generate linear samples
  (for now, only even numbers are considered, and the numbers should be sub-multiples of the loop bound).
  Args:
    loops: list containing the upper bounds of each loop
    config: configuration file
    left: indicates if lower bound should be included, i.e., 1
    right: indicates if upper bound should be included
    loop_limit: restrains the upper bound of loop factors
  """
  if config['mode'] == 'training':
    setting = config['setting']['training']
  else:
    setting = config['setting']['search']
  sample_mode = setting['sample_mode']

  # Sample each loop
  sample_list = []
  for loop in loops:
    if sample_mode == 'log':
      index = int(np.floor(np.log2(loop)))
    else:
      index = loop
    if not right:
      ub = index - 1
    else:
      ub = index
    if not left:
      if sample_mode == 'log':
        lb = 1
      else:
        lb = 2
    else:
      if sample_mode == 'log':
        lb = 0
      else:
        lb = 1
    if loop_limit != -1:
      if sample_mode == 'log':
        ub_limit = int(np.log2(loop_limit))
      else:
        ub_limit = loop_limit
      ub = min(ub, ub_limit)
    samples = range(lb, ub + 1)
#    if config['sample_mode'] == 'random':
#      if len(samples) > config['n_sample']:
#        samples = random.sample(samples, config['n_sample'])

    if sample_mode == 'log':
      samples = [np.power(2, int(s)) for s in samples]
    elif sample_mode == 'linear':
      samples = [s for s in samples if s%2 == 0 and loop%s == 0]
      if lb not in samples and loop%lb == 0:
        samples.append(lb)
      if ub not in samples and loop%ub == 0:
        samples.append(ub)
    # double check
    if not left:
      if 1 in samples:
        samples.remove(1)
    if not right:
      if loop in samples:
        samples.remove(loop)
    sample_list.append(samples)
  # Generate cartesian product
#  print("sample_list: ", sample_list)
  sample_loops = list(itertools.product(*sample_list))
  sample_loops = [list(tup) for tup in sample_loops]
#  print("sample_loops: ", sample_loops)
  if config['sample_mode'] == 'random':
    if len(sample_loops) > config['n_sample']:
      sample_loops = random.sample(sample_loops, config['n_sample'])
#      print("sample results: ", sample_loops)

  return sample_loops

def generate_simd_loop_candidates(loops, config):
  """ Generate candidate loops given the loop bounds
  This function samples each loop dimension given the sample number set in the config,
  then sets the rest dimensions as -1.
  Example: given the input loops as [4, 4], and n_sample as 2 set in config, it will first sample
  each loop dimension and generate two lists [2, 4], [2, 4]. Then, it fills the rest dimensions as -1
  and generates all candidates: [[2, -1], [4, -1], [-1, 2], [-1, 4]].
  We restrain the upper bound of SIMD factors to 16.

  Args:
    loops: list containing the upper bounds of each loop
    config: configuration file
  """
  if config['mode'] == 'training':
    setting = config['setting']['training']
    ub_limit = int(np.log2(config['setting']['training']['SIMD'][1]))
    lb_limit = int(np.log2(config['setting']['training']['SIMD'][0]))
  else:
    setting = config['setting']['search']
    ub_limit = int(np.log2(config['setting']['search']['SIMD'][1]))
#    if (config['warmup_search']):
#      lb_limit = 0
#    else:
#      lb_limit = int(np.log2(config['setting']['search']['SIMD'][0]))
    lb_limit = int(np.log2(config['setting']['search']['SIMD'][0]))
  # Sample each loop
  sample_list = []
  sample_loops = []
  for i in range(len(loops)):
    sample_loops_i = []
    loop = loops[i]
    if loop == 1:
      samples = [1]
    else:
      index = int(np.floor(np.log2(loop)))
      lb = max(0, lb_limit)
      ub = min(index, ub_limit)
      samples = range(lb, ub + 1)
      if config['sample_mode'] == 'random':
        if len(samples) > config['n_sample']:
          samples = random.sample(samples, config['n_sample'])
      samples = [np.power(2, int(s)) for s in samples]
      samples.reverse()
    for sample in samples:
      sample_loop = [-1] * len(loops)
      sample_loop[i] = sample
      sample_loops_i.append(sample_loop)
    if bool(sample_loops_i):
      sample_loops.append(sample_loops_i)

  return sample_loops

#def search_kernel(kernel_id, sa_sizes, config):
#  """ Seach for the optimal design for the exploration process
#
#  The following steps will be executed:
#  1. Compile and run top_gen file to generate the top_kernel file.
#  2. Predict the latency and resource of the design, select the optimal design by far.
#  3. (Optional) Register the design into the database.
#
#  Args:
#    kernel_id: the selected kernel id from the space_time stage
#    sa_sizes: the current sa_sizes configuration
#    config: configuration file
#  """
#  # Compile top_gen file
#  path = os.getcwd()
#  cmd = 'g++ -o polysa.tmp/src/top_gen polysa.tmp/src/kernel_top_gen.cpp - I' + path + '/isl/include -L' + path + '/isl/.libs -lisl'
#  if config['verbose']:
#    print('[PolySA Optimizer] Execute CMD: ' + cmd)
#  ret = subprocess.run(cmd.split(), stdout=config['stdout'])
#  if ret.returncode != 0:
#    print('[PolySA Optimizer] CMD: %s failed with error code: ' % (cmd) + str(ret.returncode))
#    return
#  # Run the executable
#  os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ':' + path + '/isl/.libs'
#  cmd = './polysa.tmp/src/top_gen'
#  if config['verbose']:
#    print('[PolySA Optimizer] Execute CMD: ' + cmd)
#  ret = subprocess.run(cmd.split(), stdout=config['stdout'])
#  if ret.returncode != 0:
#    print('[PolySA Optimizer] CMD: %s failed with error code: ' % (cmd) + str(ret.returncode))
#    return
#
#  # Generate the final kernel file
#  cmd = 'python3.6 ./polysa_scripts/codegen.py -c ./polysa.tmp/src/top.cpp -d ./polysa.tmp/src/kernel_kernel.cpp -p xilinx -o ./polysa.tmp/src/kernel_xilinx.cpp'
#  if config['verbose']:
#    print('[PolySA Optimizer] Execute CMD: ' + cmd)
#  ret = subprocess.run(cmd.split(), stdout=config['stdout'])
#  if ret.returncode != 0:
#    print('[PolySA Optimizer] CMD: %s failed with error code: ' % (cmd) + str(ret.returncode))
#    return
#
#  # Prepare the design files
#  # Generate kernel folder
#  if not os.path.exists('polysa.tmp/optimizer/search/kernel' + str(kernel_id)):
#    cmd = 'mkdir polysa.tmp/optimizer/search/kernel' + str(kernel_id)
#    if config['verbose']:
#      print('[PolySA Optimizer] Execute CMD: ' + cmd)
#    ret = subprocess.run(cmd.split(), stdout=config['stdout'])
#    if ret.returncode != 0:
#      print('[PolySA Optimizer] CMD: %s failed with error code: ' % (cmd) + str(ret.returncode))
#      return
#
#  # Generate sa_sizes annotation
#  sa_sizes_cmd = generate_sa_sizes_cmd(sa_sizes)
#  with open(folder_dir + '/sizes.annotation', 'w') as f:
#    f.write(sa_sizes_cmd)
#
#  # Cp files
#  cmd = 'cp -r ./polysa.tmp/latency_est ' + folder_dir + '/'
#  if config['verbose']:
#    print('[PolySA Optimizer] Execute CMD: ' + cmd)
#  ret = subprocess.run(cmd.split(), stdout=config['stdout'])
#  if ret.returncode != 0:
#    print('[PolySA Optimizer] CMD: %s failed with error code: ' % (cmd) + str(ret.returncode))
#    return
#
#  cmd = 'cp -r ./polysa.tmp/resource_est ' + folder_dir + '/'
#  if config['verbose']:
#    print('[PolySA Optimizer] Execute CMD: ' + cmd)
#  ret = subprocess.run(cmd.split(), stdout=config['stdout'])
#  if ret.returncode != 0:
#    print('[PolySA Optimizer] CMD: %s failed with error code: ' % (cmd) + str(ret.returncode))
#    return
#
#  cmd = 'cp -r ./polysa.tmp/src ' + folder_dir + '/'
#  if config['verbose']:
#    print('[PolySA Optimizer] Execute CMD: ' + cmd)
#  ret = subprocess.run(cmd.split(), stdout=config['stdout'])
#  if ret.returncode != 0:
#    print('[PolySA Optimizer] CMD: %s failed with error code: ' % (cmd) + str(ret.returncode))
#    return
#
#  cmd = 'cp ' + config['kernel_file_path'] + '/* ' + folder_dir + '/src/'
#  if config['verbose']:
#    print('[PolySA Optimizer] Execute CMD: ' + cmd)
#  ret = subprocess.Popen(cmd, shell=True, stdout=config['stdout'])
#  if ret.returncode != 0:
#    print('[PolySA Optimizer] CMD: %s failed with error code: ' % (cmd) + str(ret.returncode))
#    return
#
#  # Clear the content
#  cmd = 'rm ./polysa.tmp/latency_est/*'
#  if config['verbose']:
#    print('[PolySA Optimizer] Execute CMD: ' + cmd)
#  ret = subprocess.Popen(cmd, shell=True, stdout=config['stdout'])
#  if ret.returncode != 0:
#    print('[PolySA Optimizer] CMD: %s failed with error code: ' % (cmd) + str(ret.returncode))
#    return
#
#  cmd = 'rm ./polysa.tmp/resource_est/*'
#  if config['verbose']:
#    print('[PolySA Optmizer] Execute CMD: ' + cmd)
#  ret = subprocess.Popen(cmd, shell=True, stdout=config['stdout'])
#  if ret.returncode != 0:
#    print('[PolySA Optimizer] CMD: %s failed with error code: ' % (cmd) + str(ret.returncode))
#    return
#
#  cmd = 'rm ./polysa.tmp/src/*'
#  if config['verbose']:
#    print('[PolySA Optimizer] Execute CMD: ' + cmd)
#  ret = subprocess.Popen(cmd, shell=True, stdout=config['stdout'])
#  if ret.returncode != 0:
#    print('[PolySA Optimizer] CMD: %s failed with error code: ' % (cmd) + str(ret.returncode))
#    return
#
#  return

def save_design_files(kernel_id, sa_sizes, config):
  path = os.getcwd()
  # Prepare the design files
  # Generate kernel folder
  if config['mode'] == 'training':
    if not os.path.exists('polysa.tmp/optimizer/training/kernel' + str(kernel_id)):
      cmd = 'mkdir polysa.tmp/optimizer/training/kernel' + str(kernel_id)
      if config['verbose']:
        print('[PolySA Optimizer] Execute CMD: ' + cmd)
      ret = subprocess.run(cmd.split(), stdout=config['stdout'])

    designs = os.listdir('polysa.tmp/optimizer/training/kernel' + str(kernel_id))
    design_id = len(designs)
    folder_dir = path + '/polysa.tmp/optimizer/training/kernel' + str(kernel_id) + '/design' + str(design_id)
    cmd = "mkdir " + folder_dir
    if config['verbose']:
      print("[PolySA Optimizer] Execute CMD: " + cmd)
    ret = subprocess.run(cmd.split(), stdout=config['stdout'])
  elif config['mode'] == 'search':
    if config['work_dir'] == 'polysa.tmp':
      kernel_folder = config['work_dir'] + '/optimizer/search/kernel' + str(kernel_id)
    else:
      kernel_folder = config['work_dir'] + '/kernel' + str(kernel_id)
    if not os.path.exists(kernel_folder):
      cmd = 'mkdir ' + kernel_folder
      if config['verbose']:
        print('[PolySA Optimizer] Execute CMD: ' + cmd)
      ret = subprocess.run(cmd.split(), stdout=config['stdout'])

    designs = os.listdir(kernel_folder)
    design_id = len(designs)
    folder_dir = kernel_folder + '/design' + str(design_id)
    cmd = "mkdir " + folder_dir
    if config['verbose']:
      print("[PolySA Optimizer] Execute CMD: " + cmd)
    ret = subprocess.run(cmd.split(), stdout=config['stdout'])

  # Generate sa_sizes annotation
  sa_sizes_cmd = generate_sa_sizes_cmd(sa_sizes)
  with open(folder_dir + '/sizes.annotation', 'w') as f:
    f.write(sa_sizes_cmd)

  # Store latency and resource information
  if config['mode'] == 'search':
    est_info = {}
#    print(config['opt_latency'])
#    print(config['opt_resource'])
#    est_info['latency'] = int(config['opt_latency'][0,0])
    if type(config['opt_latency']).__module__ == np.__name__:
      est_info['latency'] = int(np.asscalar(config['opt_latency']))
    else:
      est_info['latency'] = int(config['opt_latency'])
    est_info['resource'] = {}
    for res in config['opt_resource']:
#      print(config['opt_resource'][res])
      if type(config['opt_resource'][res]).__module__ == np.__name__:
        res_usage = int(np.asscalar(config['opt_resource'][res]))
      else:
        res_usage = int(config['opt_resource'][res])
      est_info['resource'][res] = res_usage
    with open(folder_dir + '/est.json', 'w') as f:
      json.dump(est_info, f, indent=4)
#    with open(folder_dir + '/est.info', 'w') as f:
#      f.write('latency est.: ' + str(config['opt_latency']) + '\n')
#      f.write('resource est.: ' + str(config['opt_resource']) + '\n')

  # Cp files
  cmd = 'cp -r ' + config['work_dir'] + '/latency_est ' + folder_dir + '/'
  if config['verbose']:
    print('[PolySA Optimizer] Execute CMD: ' + cmd)
  ret = subprocess.run(cmd.split(), stdout=config['stdout'])

  cmd = 'cp -r ' + config['work_dir'] + '/resource_est ' + folder_dir + '/'
  if config['verbose']:
    print('[PolySA Optimizer] Execute CMD: ' + cmd)
  ret = subprocess.run(cmd.split(), stdout=config['stdout'])

  cmd = 'cp -r ' + config['work_dir'] + '/src ' + folder_dir + '/'
  if config['verbose']:
    print('[PolySA Optimizer] Execute CMD: ' + cmd)
  ret = subprocess.run(cmd.split(), stdout=config['stdout'])

  cmd = 'cp ' + config['kernel_file_path'] + '/* ' + folder_dir + '/src/'
  if config['verbose']:
    print('[PolySA Optimizer] Execute CMD: ' + cmd)
  ret = subprocess.Popen(cmd, shell=True, stdout=config['stdout'])

def clear_design_files(config):
  cmd = 'rm ' + config['work_dir'] + '/latency_est/*'
  if config['verbose']:
    print('[PolySA Optimizer] Execute CMD: ' + cmd)
  ret = subprocess.Popen(cmd, shell=True, stdout=config['stdout'], stderr=subprocess.DEVNULL)

  cmd = 'rm ' + config['work_dir'] + '/resource_est/*'
  if config['verbose']:
    print('[PolySA Optimizer] Execute CMD: ' + cmd)
  ret = subprocess.Popen(cmd, shell=True, stdout=config['stdout'], stderr=subprocess.DEVNULL)

  cmd = 'rm ' + config['work_dir'] + '/src/*'
  if config['verbose']:
    print('[PolySA Optimizer] Execute CMD: ' + cmd)
  ret = subprocess.Popen(cmd, shell=True, stdout=config['stdout'], stderr=subprocess.DEVNULL)

def explore_kernel(kernel_id, sa_sizes, config):
  """ Prepare it for the later training or search process

  By far the program should be generated under the PolySA directory, including the following files:
  - kernel_kernel.h: header file of the generated kernel
  - kernel_kernel.cpp: source file containing the definition of kernel functions
  - kernel_hls_host.cpp: the HLS host file
  - kernel_top_gen.cpp: source file to generate the top kernel file
  - kernel_top_gen.h: header file of top_gen function
  - latency_est/[module_name]_loop_info.json: the loop structure information of all modules
  - latency_est/array_info.json: the global array information of the kernel
  - resource_est/design_info.json: the design information of all modules

  The following steps will be executed:
  1. Compile and run top_gen file to generate the top_kernel file.
  2. Move the design files and latency/resource_est files to a folder
     and generate the configuration file that will be later used for training.

  Args:
    kernel_id: the selected kernel id from the space_time stage
    sa_sizes: the current sa_sizes configuration
    config: configuration file
  """
  # Compile top_gen file
  path = os.getcwd()
  cmd = 'g++ -o ' + config['work_dir'] + '/src/top_gen ' + config['work_dir'] + '/src/kernel_top_gen.cpp -I' + path + '/isl/include -L' + path + '/isl/.libs -lisl'
  if config['verbose']:
    print('[PolySA Optimizer] Execute CMD: ' + cmd)
  ret = subprocess.run(cmd.split(), stdout=config['stdout'])
  if ret.returncode != 0:
    print('[PolySA Optimizer] CMD: %s failed with error code: ' % (cmd) + str(ret.returncode))
    return
  # Run the executable
  os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ":" + path + '/isl/.libs'
  cmd = config['work_dir'] + '/src/top_gen'
  if config['verbose']:
    print('[PolySA Optimizer] Execute CMD: ' + cmd)
  ret = subprocess.run(cmd.split(), stdout=config['stdout'])
  if ret.returncode != 0:
    print('[PolySA Optimizer] CMD: %s failed with error code: ' % (cmd) + str(ret.returncode))
    return

  if config['mode'] == 'training':
    # Generate the final kernel file
    cmd = 'python3.6 ./polysa_scripts/codegen.py -c ' + config['work_dir'] + '/src/top.cpp -d ' + config['work_dir'] + '/src/kernel_kernel.cpp -p xilinx -o ' + config['work_dir'] + '/src/kernel_xilinx.cpp'
    if config['verbose']:
      print('[PolySA Optimizer] Execute CMD: ' + cmd)
    ret = subprocess.run(cmd.split(), stdout=config['stdout'])
    if ret.returncode != 0:
      print('[PolySA Optimizer] CMD: %s failed with error code: ' % (cmd) + str(ret.returncode))
      return

  if config['mode'] == 'training':
    save_design_files(kernel_id, sa_sizes, config)
  elif config['mode'] == 'search':
    # Predict latency and resource models
    design_info = res_model.extract_design_info(    \
        config['work_dir'] + '/resource_est/design_info.json', \
        config['work_dir'] + '/resource_est/design_info.dat',  \
        None)
    design_infos = [design_info]

    # Extract the latency infos
    latency_info = latency_model.extract_latency_info(  \
          config['work_dir'] + '/latency_est', \
          None)
    latency_infos = [latency_info]

    # Reorganize the design_infos into a dataframe
    ret_res = res_model.convert_design_infos_to_df(design_infos)
    # Predict resource
    start_time = time.time()
    DSP_usage = res_model.predict_kernel_DSP_usage('kernel' + str(kernel_id), ret_res['df'], ret_res['module_list'], ret_res['fifo_list'], design_info, config['verbose'])
    DSP_usage = np.asscalar(DSP_usage)
    # DSP
    if DSP_usage > config['hw_info']['DSP'] * config['setting']['search']['res_thres']['DSP'][1]:
      clear_design_files(config)
      config['monitor']['res_pruned']['DSP'] += 1
      config['monitor']['res_est_time'] += (time.time() - start_time)
      config['monitor']['res_est_n'] += 1
      return
#    if not config['warmup_search']:
#      if res_usage['DSP'] < config['hw_info']['DSP'] * config['setting']['search']['res_thres']['DSP'][0]:
#        clear_design_files(config)
#        return
    if DSP_usage < config['hw_info']['DSP'] * config['setting']['search']['res_thres']['DSP'][0]:
      config['monitor']['boundary_design']['dsp_underuse'] = 1
      clear_design_files(config)
      config['monitor']['res_pruned']['DSP'] += 1
      config['monitor']['res_est_time'] += (time.time() - start_time)
      config['monitor']['res_est_n'] += 1
      return
    config['monitor']['boundary_design']['dsp_underuse'] = 0

    # Reorganize the loop_infos into a dataframe
    ret_lat = latency_model.convert_latency_infos_to_df(latency_infos, design_infos)
    # Predict latency
    start_time = time.time()
    latency = latency_model.predict_kernel_latency('kernel' + str(kernel_id), ret_lat['df'], ret_lat['module_list'], ret_lat['stmt_list'], latency_info, config['verbose'], config['setting']['search']['cycle'], config['opt_latency'])
    config['monitor']['lat_est_time'] += (time.time() - start_time)
    config['monitor']['lat_est_n'] += 1
    config['monitor']['boundary_design']['latency'] = latency
    if config['setting']['search']['pruning'] == 1:
      if config['opt_latency'] != -1 and latency > config['opt_latency']:
        clear_design_files(config)
        config['monitor']['latency_pruned'] += 1
        return

#    res_usage = res_model.predict_kernel_resource_usage('kernel' + str(kernel_id), ret['df'], ret['module_list'], ret['fifo_list'], design_info, config['verbose'])
    start_time = time.time()
    res_usage = res_model.predict_kernel_non_DSP_usage('kernel' + str(kernel_id), ret_res['df'], ret_res['module_list'], ret_res['fifo_list'], design_info, config['verbose'])
    config['monitor']['res_est_time'] += (time.time() - start_time)
    config['monitor']['res_est_n'] += 1

    res_usage['FF'] = np.asscalar(res_usage['FF'])
    res_usage['LUT'] = np.asscalar(res_usage['LUT'])
    res_usage['DSP'] = DSP_usage

    FF_usage = res_usage['FF']
    LUT_usage = res_usage['LUT']
    BRAM_usage = res_usage['BRAM']
    URAM_usage = res_usage['URAM']

    # BRAM
    if BRAM_usage > config['hw_info']['BRAM'] * config['setting']['search']['res_thres']['BRAM'][1]:
      clear_design_files(config)
      config['monitor']['res_pruned']['BRAM'] += 1
      return
    if not config['warmup_search']:
      if BRAM_usage < config['hw_info']['BRAM'] * config['setting']['search']['res_thres']['BRAM'][0]:
        clear_design_files(config)
        config['monitor']['res_pruned']['BRAM'] += 1
        return

    # FF
    if FF_usage > config['hw_info']['FF'] * config['setting']['search']['res_thres']['FF'][1]:
      clear_design_files(config)
      config['monitor']['res_pruned']['FF'] += 1
      return
    if not config['warmup_search']:
      if FF_usage < config['hw_info']['FF'] * config['setting']['search']['res_thres']['FF'][0]:
        clear_design_files(config)
        config['monitor']['res_pruned']['FF'] += 1
        return

    # LUT
    if LUT_usage > config['hw_info']['LUT'] * config['setting']['search']['res_thres']['LUT'][1]:
      clear_design_files(config)
      config['monitor']['res_pruned']['LUT'] += 1
      return
    if not config['warmup_search']:
      if LUT_usage < config['hw_info']['LUT'] * config['setting']['search']['res_thres']['LUT'][0]:
        clear_design_files(config)
        config['monitor']['res_pruned']['LUT'] += 1
        return

    # URAM
    if config['hw_info']['URAM'] != 0:
      if URAM_usage > config['hw_info']['URAM'] * config['setting']['search']['res_thres']['URAM'][1]:
        clear_design_files(config)
        config['monitor']['res_pruned']['URAM'] += 1
        return
      if not config['warmup_search']:
        if URAM_usage < config['hw_info']['URAM'] * config['setting']['search']['res_thres']['URAM'][0]:
          clear_design_files(config)
          config['monitor']['res_pruned']['URAM'] += 1
          return

    if latency == config['opt_latency']:
      # We compare if the new design saves resource, this is done by caculating a mean of all resource usage.
      res_ratio = [0.1 * float(FF_usage) / config['hw_info']['FF'],     \
                   0.2 * float(LUT_usage) / config['hw_info']['LUT'],   \
                   0.3 * float(BRAM_usage) / config['hw_info']['BRAM'], \
                   0.3 * float(DSP_usage) / config['hw_info']['DSP']]
      if config['hw_info']['URAM'] != 0:
        res_ratio.append(0.1 * float(res_usage['URAM']) / config['hw_info']['URAM'])
      res_score = mean(res_ratio)

      opt_res_ratio = [0.1 * float(config['opt_resource']['FF']) / config['hw_info']['FF'],     \
                       0.2 * float(config['opt_resource']['LUT']) / config['hw_info']['LUT'],   \
                       0.3 * float(config['opt_resource']['BRAM']) / config['hw_info']['BRAM'], \
                       0.3 * float(config['opt_resource']['DSP']) / config['hw_info']['DSP']]
      if config['hw_info']['URAM'] != 0:
        opt_res_ratio.append(0.1 * float(config['opt_resource']['URAM']) / config['hw_info']['URAM'])
      opt_res_score = mean(opt_res_ratio)
      if res_score >= opt_res_score:
        clear_design_files(config)
        return

    config['monitor']['design_cnt'] += 1
    config['opt_latency'] = latency
    config['opt_resource'] = res_usage
    # register the design
    save_design_files(kernel_id, sa_sizes, config)

  clear_design_files(config)

def call_explore_kernel(kernel_id, sa_sizes, cmds, simd_info, config):
  sa_sizes_cmd = generate_sa_sizes_cmd(sa_sizes)
  cmds[-1] = sa_sizes_cmd
  cmd = ' '.join(cmds)
  # Execute the cmd
  if config['verbose']:
    print('[PolySA Optimizer] Execute CMD: ' + cmd)
  proc = subprocess.Popen(cmd.split(), stdin=subprocess.PIPE, stdout=config['stdout'])
  for loop_i in simd_info:
    proc.communicate(str.encode(loop_i + '\n'))
  if proc.returncode != 0:
    print("[PolySA Optimizer] CMD: %s failed with error code: " % (cmd) + str(proc.returncode))
    return
#    # If the kernel is failed, we will skip this example and continue
#    return
  explore_kernel(kernel_id, sa_sizes, config)

def explore_simd(kernel_id, loops, cmds, sa_sizes, config):
  """ Generate simd training candidates and proceed to training stages

  Args:
    kernel_id: the selected kernel id from the space_time stage
    loops: list containing the upper bounds of candidate loops
    cmd_prefix: the current cmd prefix
    sa_sizes: the current sa_sizes configuration
    config: configuration file
  """
  kernel_name = 'kernel' + str(kernel_id)
  simd_info = config['simd_info'][kernel_name]['reduction']

  # Generate a set of uniformly distributed tiling factors to proceed
#  print(loops)
  simd_loops_pool = generate_simd_loop_candidates(loops, config)
  if len(simd_loops_pool) == 0:
    # No available tiling options, we will disable this step and skip it
    config['polysa_config']['simd']['enable'] = 0
    with open(config['work_dir'] + '/polysa_config.json', 'w') as f:
      json.dump(config['polysa_config'], f, indent=4)
    # Update sizes
    new_sa_sizes = sa_sizes.copy()
    # Call kernel
    call_explore_kernel(kernel_id, new_sa_sizes, cmds, simd_info, config)
    # Revert back the polysa_config for the next exploration
    config['polysa_config']['simd']['enable'] = 1
    with open(config['work_dir'] + '/polysa_config.json', 'w') as f:
      json.dump(config['polysa_config'], f, indent=4)
  else:
    for loop_list in simd_loops_pool:
      for loop in loop_list:

        # Update sizes
        new_sa_sizes = sa_sizes.copy()
        new_sa_sizes.append('kernel[0]->simd' + str(loop).replace(' ', ''))
        # call kernel
        call_explore_kernel(kernel_id, new_sa_sizes, cmds, simd_info, config)
        if config['mode'] == 'search':
          if config['setting']['search']['pruning']:
            # Pruning: Since the simd factor is sorted in decreasing order, if the current
            # SIMD factor leads to longer latency, we could safely prune away the rest of the
            # candidates.
            if config['monitor']['boundary_design']['latency'] != -1:
              if config['monitor']['boundary_design']['latency'] > config['opt_latency']:
                break
            if config['monitor']['boundary_design']['dsp_underuse'] == 1:
              config['monitor']['boundary_design']['dsp_underuse'] = 0
              break

def call_explore_simd(kernel_id, sa_sizes, cmds, simd_info, config):
  sa_sizes_cmd = generate_sa_sizes_cmd(sa_sizes)
  cmds[-1] = sa_sizes_cmd
  cmd = ' '.join(cmds)
  # Execute the cmd
  if config['verbose']:
    print("[PolySA Optimizer] Execute CMD: " + cmd)
  proc = subprocess.Popen(cmd.split(), stdin=subprocess.PIPE, stdout=config['stdout'])
  for loop_i in simd_info:
    # proc.communicate(str.encode('y\n'))
    proc.communicate(str.encode(loop_i + '\n'))
  if proc.returncode != 0:
    print('[PolySA Optimizer] CMD: %s failed with error code: ' % (cmd) + str(proc.returncode))
    return
  # The program will terminate after the SIMD vectorization
  # Fetch the tuning info
  with open(config['work_dir'] + '/tuning.json') as f:
    tuning = json.load(f)
#  if 'simd' not in tuning:
#    print(tuning)
#    print(sa_sizes)
#    print(cmds)
#    print(config['work_dir'])
  if 'simd' not in tuning:
    clear_design_files(config)
#    print('returned from SIMD')
    return
  loops = tuning['simd']['tilable_loops']
  # Compute the total PE numbers.
  sa_dims = tuning['simd']['sa_dims']
  n_pes = 1
  for dim in sa_dims:
    n_pes *= int(dim)
  if config['mode'] == 'training':
    PE_ub = config['setting']['training']['PE'][1]
    PE_lb = config['setting']['training']['PE'][0]
  elif config['mode'] == 'search':
    PE_ub = config['setting']['search']['PE'][1]
#    if config['warmup_search']:
#      PE_lb = 1
#    else:
#      PE_lb = config['setting']['search']['PE'][0]
    PE_lb = config['setting']['search']['PE'][0]
#  print("PEs", n_pes)
  if n_pes > PE_ub:
    if config['verbose']:
      print('[PolySA Optimizer] Error: #PE %d exceeds the bound %d. Abort!' % (n_pes, PE_ub))
    clear_design_files(config)
    return
  if n_pes < PE_lb:
    if config['verbose']:
      print('[PolySA Optimizer] Error: #PE %d below the bound %d. Abort!' % (n_pes, PE_lb))
    clear_design_files(config)
    return
  if config['mode'] == 'search' and config['warmup_search'] == 0:
    if len(sa_dims) == 2:
      sa_dims.sort(reverse=True)
      pe_ratio = sa_dims[0] / sa_dims[1]
      if pe_ratio > config['setting']['search']['PE_ratio']:
        if config['verbose']:
          print('[PolySA Optimizer] Error: PE ratio %d exceeds the bound %d. Abort!' % (pe_ratio, config['setting']['PE_ratio']))

  explore_simd(kernel_id, loops, cmds, sa_sizes, config)

def explore_latency_hiding(kernel_id, loops, cmds, sa_sizes, config):
  """ Generate latency hiding training candidates and proceed to simd vectorization
  We restrain the upper bounds of latency tiling factors to 64.

  Args:
    kernel_id: the selected kernel id from the space_time stage
    loops: list containing the upper bounds of candidate loops
    cmd_prefix: the current cmd prefix
    sa_sizes: the current sa_sizes configuration
    config: configuration file
  """
  kernel_name = 'kernel' + str(kernel_id)
  if kernel_name not in config['simd_info']:
    print("[PolySA Optimizer] Error: Please provide SIMD information for " + kernel_name)
    sys.exit()
  simd_info = config['simd_info'][kernel_name]['reduction']


  # Generate a set of uniformly distributed tiling factors to proceed
  latency_loops_pool = generate_loop_candidates(loops, config, 1, 0, 64)

  if len(latency_loops_pool) == 0:
    # No available tiling options, we will disable this step and skip it
    config['polysa_config']['latency']['enable'] = 0
    with open(config['work_dir'] + '/polysa_config.json', 'w') as f:
      json.dump(config['polysa_config'], f, indent=4)
    # Update the sizes
    new_sa_sizes = sa_sizes.copy()
    # Call simd
#    if not config:
#      print('config not found')
    call_explore_simd(kernel_id, new_sa_sizes, cmds, simd_info, config)
    # Revert the changes
    config['polysa_config']['latency']['enable'] = 1
    with open(config['work_dir'] + '/polysa_config.json', 'w') as f:
      json.dump(config['polysa_config'], f, indent=4)
  else:
    for loop in latency_loops_pool:

 #  for loop in [latency_loops_pool[2]]:
      # Compute the latency hiding size
      lat_hiding_len = 1
      for tile_size in loop:
        lat_hiding_len *= tile_size
#      print("Latency: ", lat_hiding_len)
      if config['mode'] == 'search':
        if config['setting']['search']['pruning'] == 1 and not config['warmup_search']:
          if lat_hiding_len > config['setting']['search']['latency_hiding'][1] or \
             lat_hiding_len < config['setting']['search']['latency_hiding'][0]:
            continue

      # Update the sizes
      new_sa_sizes = sa_sizes.copy()
      new_sa_sizes.append('kernel[0]->latency' + str(loop).replace(' ', ''))
      # Call simd
      call_explore_simd(kernel_id, new_sa_sizes, cmds, simd_info, config)

def call_explore_latency(kernel_id, sa_sizes, cmds, config):
  sa_sizes_cmd = generate_sa_sizes_cmd(sa_sizes)
  cmds[-1] = sa_sizes_cmd
  cmd = ' '.join(cmds)
  # Execute the cmd
  if config['verbose']:
    print("[PolySA Optimizer] Execute CMD: " + cmd)
  ret = subprocess.run(cmd.split(), stdout=config['stdout'])
  if ret.returncode != 0:
    print("[PolySA Optimizer] CMD: %s failed with error code: " % (cmd) + str(ret.returncode))
    return
  # The program will terminate after the latency hiding
  # Fetch the tuning info
  with open (config['work_dir'] + '/tuning.json') as f:
    tuning = json.load(f)
  loops = tuning['latency']['tilable_loops']
  explore_latency_hiding(kernel_id, loops, cmds, sa_sizes, config)

def explore_array_part_L2(kernel_id, loops, coincident, cmds, sa_sizes, config):
  """ Generate L2 array partitioning candidates

  As a heuristic, we only apply L2 array partitioning on parallel loops to save off-chip
  communication. We examine from outer loops to inner loops. Once we meet a non-parallel
  loop, we will stop from here, and set the tile factors from here to below to maximum.
  """
  # Apply heuristic
  loop_end = 0
  for loop_id in range(len(loops)):
    if not coincident[loop_id]:
      loop_end = loop_id
      break
  loops_interest = loops[:loop_end]
  array_part_L2_loops_pool = generate_loop_candidates(loops_interest, config, 1, 1)
  array_part_L2_loops_pool = [loop + loops[loop_end:] for loop in array_part_L2_loops_pool]

#  print("input: ", loops)
#  print("output: ", array_part_L2_loops_pool)
#  array_part_L2_loops_pool = generate_loop_candidates(loops, config, 1, 1)
  if len(array_part_L2_loops_pool) == 0:
    # No available tiling optioons, we will disable this step and skip it
    config['polysa_config']['array_part_L2']['enable'] = 0
    with open(config['work_dir'] + '/polysa_config.json', 'w') as f:
      json.dump(config['polysa_config'], f, indent=4)
    # Update the sizes
    new_sa_sizes = sa_sizes.copy()
    # Call latency
    call_explore_latency(kernel_id, new_sa_sizes, cmds, config)
    # Revert the changes
    config['polysa_config']['array_part_L2']['enable'] = 1
    with open(config['work_dir'] + '/polysa_config.json', 'w') as f:
      json.dump(config['polysa_config'], f, indent=4)
  else:
    for loop in array_part_L2_loops_pool:
      # Update the sizes
      new_sa_sizes = sa_sizes.copy()
      new_sa_sizes.append('kernel[0]->array_part_L2' + str(loop).replace(' ', ''))
      # Call simd
      call_explore_latency(kernel_id, new_sa_sizes, cmds, config)

def call_explore_array_part_L2(kernel_id, sa_sizes, cmds, config):
  sa_sizes_cmd = generate_sa_sizes_cmd(sa_sizes)
  cmds[-1] = sa_sizes_cmd
  cmd = ' '.join(cmds)
  # Execute the cmd
  if config['verbose']:
    print("[PolySA Optimizer] Execute CMD: " + cmd)
  ret = subprocess.run(cmd.split(), stdout=config['stdout'])
  if ret.returncode != 0:
    print("[PolySA Optimizer] CMD: %s failed with error code: " % (cmd) + str(ret.returncode))
    return
  # The program will terminate after the L2 array partitioning
  # Fetch the tuning info
#  print(config['work_dir'])
  with open(config['work_dir'] + '/tuning.json') as f:
    tuning = json.load(f)
  loops = tuning['array_part_L2']['tilable_loops']
  coincident = tuning['array_part_L2']['coincident']
  explore_array_part_L2(kernel_id, loops, coincident, cmds, sa_sizes, config)

def explore_array_part_single_job(loops, kernel_id, sa_sizes, cmds, config, work_dir):
  # Modify the cmds
  cmds[1] = '--config=' + work_dir + '/polysa_config.json'
  cmds[2] = '--output-dir=' + work_dir

  if config['mode'] == 'search' and config['setting']['search']['n_job'] > 1:
    config['verbose'] = 0
  if config['verbose']:
    config['stdout'] = None
  else:
    config['stdout'] = subprocess.DEVNULL

  config['work_dir'] = work_dir
#  cmd_prefix = cmd_prefix + ' --config=' + work_dir + '/polysa_config.json'

  set_meter = 0
  if config['mode'] == 'search' and config['setting']['search']['n_job'] == 1:
    # Set up progress meter
    total_cnt = len(loops)
    visited_cnt = 0
    set_meter = 1
  for loop in loops:
    if set_meter:
      visited_cnt += 1
      elapsed_time = time.time() - config['monitor']['time']
      print('[PolySA Optimizer] Progress estimate: %.1f%% Elapsed time(s): %d' % \
          (float(visited_cnt)/total_cnt*100, elapsed_time))
      print('[PolySA Optimizer] Latency pruned: %d' % (config['monitor']['latency_pruned']))
      print('[PolySA Optimizer] FF pruned: %d LUT pruned: %d DSP pruned: %d BRAM pruned: %d URAM pruned: %d' % \
              (config['monitor']['res_pruned']['FF'], \
               config['monitor']['res_pruned']['LUT'], config['monitor']['res_pruned']['DSP'], \
               config['monitor']['res_pruned']['BRAM'], config['monitor']['res_pruned']['URAM']))
      if config['monitor']['lat_est_n'] > 1:
        print('[PolySA Optimizer] Latency est. average time: %.4f' % \
            (config['monitor']['lat_est_time'] / config['monitor']['lat_est_n']))
      if config['monitor']['res_est_n'] > 1:
        print('[PolySA Optimizer] Res. est. average time: %.4f' % \
            (config['monitor']['res_est_time'] / config['monitor']['res_est_n']))

    # Update the sizes
    new_sa_sizes = sa_sizes.copy()
    new_sa_sizes.append('kernel[0]->array_part' + str(loop).replace(' ', ''))
    if config['two_level_buffer'] == 1:
      # Call array_part_L2
      call_explore_array_part_L2(kernel_id, new_sa_sizes, cmds, config)
    else:
      # Call latency
      call_explore_latency(kernel_id, new_sa_sizes, cmds, config)
  return config

def explore_array_part(kernel_id, cmds, config):
  """ Generate array partitioning training candidates and proceed to latency hiding

  We apply the following heuristic to prune the array_part candidate loops.
  - The product of tiling factors should be no less than than the PE_lb.

  Args:
    kernel_id: the selected kernel_id from the space_time stage
    cmds: the list containing cmds
          - cmds[0]: the original user command
          - cmds[1]: the polysa config file
          - cmds[2]: the polysa ouput dir
          - cmds[3]: the polysa sizes
    config: tuning configuration
  """
  # Update the cmd
  sa_sizes = ['kernel[0]->space_time[' + str(kernel_id) + ']']
  sa_sizes_cmd = generate_sa_sizes_cmd(sa_sizes)
  cmds.append(sa_sizes_cmd)
  cmd = ' '.join(cmds)
  # Execute the cmd
  if config['verbose']:
    print("[PolySA Optimizer] Execute CMD: " + cmd)
  ret = subprocess.run(cmd.split(), stdout=config['stdout'])
  if ret.returncode != 0:
    print("[PolySA Optimizer] CMD: %s failed with error code: " % (cmd) + str(ret.returncode))
    return
  # The program will terminate after the array partitioning
  # Fetch the tuning info
  with open('polysa.tmp/tuning.json') as f:
    tuning = json.load(f)
  loops = tuning['array_part']['tilable_loops']
  # Generate a set of uniformly distributed tiling factors to proceed
  array_part_loops_pool = generate_loop_candidates(loops, config, 0, 1)
##debug
#  for loop in array_part_loops_pool:
#    if loop == [128, 160, 256]:
#      print('find it in array_part_loops')
#  array_part_loops_pool = [[128, 160, 256]]

  # Apply heuristics to filter out the array_part_loops
  if config['mode'] == 'search' and config['warmup_search'] == 0:
    pruned_array_part_loops_pool = []
    PE_lb = config['setting']['search']['PE'][0]
    PE_ub = config['setting']['search']['PE'][1]
    n_sa_dim = tuning['array_part']['n_sa_dim']
    for array_part_loop in array_part_loops_pool:
      prod = 1
      for loop in array_part_loop:
        if loop > 1:
          prod *= loop
      if prod < PE_lb:
        continue
#      prod_ub = PE_ub
#      for loop_id in range(n_sa_dim, len(array_part_loop)):
#        if array_part_loop[loop_id] > 1:
#          prod_ub *= array_part_loop[loop_id]
#      if prod > prod_ub:
#        continue
      pruned_array_part_loops_pool.append(array_part_loop)
    array_part_loops_pool = pruned_array_part_loops_pool
#    print("pruned: ", len(array_part_loops_pool))
##debug
#  for loop in array_part_loops_pool:
#    if loop == [128, 160, 256]:
#      print('find it in array_part_loops')

  if len(array_part_loops_pool) == 0:
    # No available tiling options, we will disable the step and skip it.
    # At the same time, two_level_buffer is disabled
    config['polysa_config']['array_part']['enable'] = 0
    config['polysa_config']['array_part_L2']['enable'] = 0
    with open('polysa.tmp/polysa_config.json', 'w') as f:
      json.dump(config['polysa_config'], f, indent=4)
    config['work_dir'] = 'polysa.tmp'
    # Update the sizes
    new_sa_sizes = sa_sizes.copy()
    # Call latency
    call_explore_latency(kernel_id, new_sa_sizes, cmd_prefix, config)
    # Revert the changes
    config['polysa_config']['array_part']['enable'] = 1
    config['polysa_config']['array_part_L2']['enable'] = 1
    with open('polysa.tmp/polysa_config.json', 'w') as f:
      json.dump(config['polysa_config'], f, indent=4)
  else:
    # Randomly shuffle the pool
#    print(array_part_loops_pool)
    random.shuffle(array_part_loops_pool)
#    print(array_part_loops_pool)
    if config['mode'] == 'search' and config['setting']['search']['n_job'] > 1:
      # multi-process parallelization
      num_proc = min(multiprocessing.cpu_count(), config['setting']['search']['n_job'])
      # split the array_part_loops into chunks
      chunk_size = int(np.ceil(float(len(array_part_loops_pool)) / num_proc))
      loop_chunks = [array_part_loops_pool[i:i + min(chunk_size, len(array_part_loops_pool) - i)] for i in range(0, len(array_part_loops_pool), chunk_size)]
      pool = multiprocessing.Pool(processes=num_proc)
      # Allocate the folders and copy the files
      for i in range(num_proc):
        prj_dir = './polysa.tmp/optimizer/search/job' + str(i)
        if not os.path.exists(prj_dir):
          sys_cmd = "mkdir " + prj_dir
          if config['verbose']:
            print('[PolySA Optimizer] Execute CMD: ' + sys_cmd)
          ret = subprocess.run(sys_cmd.split(), stdout=config['stdout'])
          if ret.returncode != 0:
            print("[PolySA Optimizer] CMD: %s failed with error code: " % (sys_cmd) + str(ret.returncode))
            return

          sys_cmd = 'mkdir ' + prj_dir + '/latency_est'
          if config['verbose']:
            print('[PolySA Optimizer] Execute CMD: ' + sys_cmd)
          ret = subprocess.run(sys_cmd.split(), stdout=config['stdout'])
          if ret.returncode != 0:
            print("[PolySA Optimizer] CMD: %s failed with error code: " % (sys_cmd) + str(ret.returncode))
            return

          sys_cmd = 'mkdir ' + prj_dir + '/resource_est'
          if config['verbose']:
            print('[PolySA Optimizer] Execute CMD: ' + sys_cmd)
          ret = subprocess.run(sys_cmd.split(), stdout=config['stdout'])
          if ret.returncode != 0:
            print("[PolySA Optimizer] CMD: %s failed with error code: " % (sys_cmd) + str(ret.returncode))
            return

          sys_cmd = 'mkdir ' + prj_dir + '/src'
          if config['verbose']:
            print('[PolySA Optimizer] Execute CMD: ' + sys_cmd)
          ret = subprocess.run(sys_cmd.split(), stdout=config['stdout'])
          if ret.returncode != 0:
            print("[PolySA Optimizer] CMD: %s failed with error code: " % (sys_cmd) + str(ret.returncode))
            return

          sys_cmd = 'cp ./polysa.tmp/polysa_config.json ' + prj_dir + '/'
          if config['verbose']:
            print('[PolySA Optimizer] Execute CMD: ' + sys_cmd)
          ret = subprocess.run(sys_cmd.split(), stdout=config['stdout'])
          if ret.returncode != 0:
            print("[PolySA Optimizer] CMD: %s failed with error code: " % (sys_cmd) + str(ret.returncode))
            return

      work_dir = 'polysa.tmp/optimizer/search/job'
      print('[PolySA Optimizer] Forking %d subprocesses' % (num_proc))
      results = pool.starmap(explore_array_part_single_job, [(loop_chunks[i], kernel_id, sa_sizes, cmds, copy.deepcopy(config), work_dir+str(i)) for i in range(len(loop_chunks))])
      print('[PolySA Optimizer] Multi-processing finished')
      for result in results:
        print(result['opt_latency'])
        if config['opt_latency'] == -1 or \
            (result['opt_latency'] != -1 and result['opt_latency'] < config['opt_latency']):
          config['opt_latency'] = result['opt_latency']
          config['opt_resource'] = result['opt_resource']
    else:
      explore_array_part_single_job(array_part_loops_pool, kernel_id, sa_sizes, cmds, config, 'polysa.tmp')

def synth_train_samples_single_job(designs, kernel, config):
  if config['setting']['training']['n_job'] > 1:
    config['verbose'] = 0
  if config['verbose']:
    config['stdout'] = None
  else:
    config['stdout'] = subprocess.DEVNULL

  for design in designs:
    prj_dir = 'polysa.tmp/optimizer/training/' + kernel + '/' + design
    # cp tcl to prj folder
    cmd = 'cp polysa_scripts/script.tcl ' + prj_dir + '/'
    if config['verbose']:
      print('[PolySA Optimizer] Execute CMD: ' + cmd)
    ret = subprocess.run(cmd.split(), stdout=config['stdout'])

    # Execute the tcl
    cwd = os.getcwd()
    os.chdir(prj_dir)

    cmd = 'vivado_hls -f script.tcl'
    if config['verbose']:
      print('[PolySA Optimizer] Execute cmd: ' + cmd)
    ret = subprocess.run(cmd.split(), stdout=config['stdout'])

    os.chdir(cwd)

def synth_train_samples(config):
#  # Set up the environment
#  cmd = 'source /opt/tools/xilinx/Vitis/2019.2/settings64.sh'
#  print('Execute cmd: ' + cmd)
#  ret = subprocess.run(cmd)

  # Copy the script.tcl to each training folder and execute the program
  kernels = os.listdir('polysa.tmp/optimizer/training')
  kernels = sorted(kernels, key = lambda x:int(x[6:]))
  n_job = config['setting']['training']['n_job']
  for kernel in kernels:
    designs = os.listdir('polysa.tmp/optimizer/training/' + kernel)
    designs = sorted(designs, key = lambda x:int(x[6:]))
    if n_job > 1:
      # Multi-process parallelization
      num_proc = min(multiprocessing.cpu_count(), n_job)
      # Split the designs into chunks
      chunk_size = int(np.ceil(float(len(designs)) / num_proc))
      design_chunks = [designs[i:i + min(chunk_size, len(designs) - i)] for i in range(0, len(designs), chunk_size)]
      pool = multiprocessing.Pool(processes=num_proc)
      print('[PolySA Optimizer] Forking %d subprocesses' % (num_proc))
      pool.starmap(synth_train_samples_single_job, [(design_chunks[i], kernel, config) for i in range(len(design_chunks))])
      print('[PolySA Optimizer] Multi-processing finished')
    else:
      synth_train_samples_single_job(designs, kernel, config)

def train_resource_models_xilinx(config):
  """ Train the resource models for Xilinx platforms

  Args:
    config: global parameters
  """
  kernels = os.listdir('polysa.tmp/optimizer/training')
  kernels = sorted(kernels, key = lambda x:int(x[6:]))
  for kernel in kernels:
    print('[PolySA Optimizer] Train resource models for ' + kernel)
    designs = os.listdir('polysa.tmp/optimizer/training/' + kernel)
    if 'resource_models' in designs:
      designs.remove('resource_models')
    if 'latency_models' in designs:
      designs.remove('latency_models')
    designs = sorted(designs, key = lambda x:int(x[6:]))
    design_infos = []
    print('[PolySA Optimizer] Extract design information')
    # Extract design info of each HLS design
    for design in designs:
      prj_dir = 'polysa.tmp/optimizer/training/' + kernel + '/' + design
      # Parse the HLS report to extract the resource information
      design_info = res_model.extract_design_info(    \
          prj_dir + '/resource_est/design_info.json', \
          prj_dir + '/resource_est/design_info.dat',  \
          prj_dir + '/prj')
      design_infos.append(design_info)
    # Reorganize the design_infos into a dataframe
    print('[PolySA Optimizer] Pre-process datasets')
#    print(design_infos[26])
    ret = res_model.convert_design_infos_to_df(design_infos)
    # Train for each module and store the trained model
    module_list = ret['module_list']
    fifo_list = ret['fifo_list']
    df = ret['df']
    print('[PolySA Optimizer] Model training and validation')
    res_model.train_module_resource_models(kernel, df, module_list, fifo_list, design_infos, config['verbose'])

def train_latency_models_xilinx(config):
  """ Train the latency models for Xilinx platform

  Args:
    config: global parameters
  """
  kernels = os.listdir('polysa.tmp/optimizer/training')
  kernels = sorted(kernels, key = lambda x:int(x[6:]))
  for kernel in kernels:
    print('[PolySA Optimizer] Train latency models for ' + kernel)
    designs = list(os.listdir('polysa.tmp/optimizer/training/' + kernel))
    if 'resource_models' in designs:
      designs.remove('resource_models')
    if 'latency_models' in designs:
      designs.remove('latency_models')
    designs = sorted(designs, key = lambda x:int(x[6:]))
    latency_infos = []
    design_infos = []
    # Extract loop info of each HLS design
#    print(designs)
    for design in designs:
      prj_dir = 'polysa.tmp/optimizer/training/' + kernel + '/' + design
      latency_info = latency_model.extract_latency_info(  \
          prj_dir + '/latency_est', \
          prj_dir + '/prj')
      latency_infos.append(latency_info)
#      if design == 'design5':
#        print(ET.tostring(latency_info['hls_rpts']['PE'], encoding='utf8').decode('utf8'))

    # Extract design info of each HLS design
    for design in designs:
      prj_dir = 'polysa.tmp/optimizer/training/' + kernel + '/' + design
      # Parse the HLS report to extract the resource information
      design_info = res_model.extract_design_info(    \
          prj_dir + '/resource_est/design_info.json', \
          prj_dir + '/resource_est/design_info.dat',  \
          prj_dir + '/prj')
      design_infos.append(design_info)

    # Reorganize the loop_infos into a dataframe
    ret = latency_model.convert_latency_infos_to_df(latency_infos, design_infos)
    # Train for each statement and store the trained model
    latency_model.train_stmt_latency_models(kernel, ret['df'], ret['module_list'], ret['stmt_list'], config['verbose'])

def explore_space_time(cmd, config):
  """ Explore the stage of space time transformation

  Args:
    cmd: input user command
    config: global configuration
  """
  # Execute the cmd
  cmds = [cmd]
  cmds.append('--config=polysa.tmp/polysa_config.json')
  cmds.append('--output-dir=polysa.tmp')
  new_cmd = ' '.join(cmds)
  if config['verbose']:
    print("[PolySA Optimizer] Execute CMD: " + new_cmd)
  ret = subprocess.run(new_cmd.split(), stdout=config['stdout'])
  if ret.returncode != 0:
    print("[PolySA Optimizer] CMD: %s failed with error code: " % (new_cmd) + str(ret.returncode))
    return
  # The program will terminate after the space-time transformation.
  # Fetch the tuning info
  with open('polysa.tmp/tuning.json') as f:
    tuning = json.load(f)
  n_kernel = tuning['space_time']['n_kernel']

  # Iterate through different kernels
  # TODO: temporarily commented out for debugging
#  for kernel_id in range(n_kernel):
  for kernel_id in range(3, 4):
    print("[PolySA Optimizer] Search kernel" + str(kernel_id))
    explore_array_part(kernel_id, cmds, config)

def train_xilinx(cmd, config):
  """ Train the resource and latency models on Xilinx platforms

  This function first creates training design samples by uniformly
  sampling all design points.
  Then it calls Vivado HLS to synthesize all designs.
  Next it collects the results and train the resource and latency
  models using linear regression.

  Args:
    cmd: input user command
    config: global configuration
  """
  config['mode'] = 'training'
  polysa_config = config['polysa_config']
  if config['setting']['training']['n_sample'] == -1:
    config['sample_mode'] = 'exhaustive'
  else:
    config['sample_mode'] = 'random'
  config['n_sample'] = config['setting']['training']['n_sample']

  # Allocate the directory for training files
  sys_cmd = "rm -rf ./polysa.tmp/optimizer/training"
  if config['verbose']:
    print("[PolySA Optimizer] Execute CMD: " + sys_cmd)
  ret = subprocess.run(sys_cmd.split())
  sys_cmd = "mkdir ./polysa.tmp/optimizer/training"
  if config['verbose']:
    print("[PolySA Optimizer] Execute CMD: " + sys_cmd)
  ret = subprocess.run(sys_cmd.split())

  # Start the exploration with the space_time stage
  start_time = time.time()
  print("[PolySA Optimizer] Generate training sample designs...")
  explore_space_time(cmd, config)

  # Execute the HLS program to synthesize all the program
  print("[PolySA Optimizer] Synthesize training sample designs...")
  synth_train_samples(config)
  elapsed_time = time.time() - start_time
  print('[PolySA Optimizer] Elapsed time(s): %d' % (elapsed_time))

  start_time = time.time()
  # Train the linear regression models for FF, LUT, and DSPs
  # For BRAM, we use the static analysis
  train_resource_models_xilinx(config)
  # Train latency models
  train_latency_models_xilinx(config)
  elapsed_time = time.time() - start_time
  print('[PolySA Optimizer] Elapsed time(s): %d' % (elapsed_time))

def add_cycle_dse_info(prj_dir, design_infos):
  kernels = os.listdir(prj_dir)
  for kernel in kernels:
    if kernel.startswith('kernel'):
      designs = os.listdir(prj_dir + '/' + kernel)
      for design in designs:
        folder_dir = prj_dir + '/' + kernel + '/' + design
        with open(folder_dir + '/est.json') as f:
          est_info = json.load(f)
        design_info = {}
        design_info['latency'] = est_info['latency']
        design_info['resource'] = est_info['resource']
        design_info['dir'] = folder_dir
        design_infos.append(design_info)

def extract_cycle_dse_results(config):
  """ Extract the top designs from cycle DSE

  Args:
    config: global configuration
  """
  design_infos = []
  if config['setting']['search']['n_job'] == 1:
    prj_dir = 'polysa.tmp/optimizer/search'
    add_cycle_dse_info(prj_dir, design_infos)
  else:
    n_job = config['setting']['search']['n_job']
    prj_dir = 'polysa.tmp/optimizer/search'
    for job in range(n_job):
      job_prj_dir = prj_dir + '/job'
      job_prj_dir += str(job)
      add_cycle_dse_info(job_prj_dir, design_infos)
#  print(design_infos)
  # Sort designs by latency
  design_infos.sort(key=lambda x:x['latency'], reverse=False)
  # Extract the top designs
  if len(design_infos) > config['setting']['search']['fre_n_sample']:
    design_infos = design_infos[0:config['setting']['search']['fre_n_sample']]
  # Copy the designs to 'cycle_dse_results'
  idx = 0
  for design in design_infos:
    from_prj_dir = design['dir']
    to_prj_dir = 'polysa.tmp/optimizer/search/cycle_dse_results/design' + str(idx)
    sys_cmd = 'mkdir ' + to_prj_dir
    if config['verbose']:
      print('[PolySA Optimizer] Execute CMD: ' + sys_cmd)
    ret = subprocess.Popen(sys_cmd, shell=True, stdout=config['stdout'])

    sys_cmd = 'cp -r ' + from_prj_dir + '/* ' + to_prj_dir + '/'
    if config['verbose']:
      print("[PolySA Optimizer] Execute CMD: " + sys_cmd)
    ret = subprocess.Popen(sys_cmd, shell=True, stdout=config['stdout'])
    idx += 1

def search_xilinx(cmd, config):
  """ Design space exploration on Xilinx platforms

  Args:
    cmd: user cmd
    config: global configuration
            - setting: optimizer setting
              - training: {n_sample: [], n_job: []}
              - search: {n_sample: [], cycle: [], pruning: [], n_job: [], random_init: []}
            - verbose
            - stdout: None/suprocess.DEVNULL
            - two_level_buffer
            - polysa_config
            - kernel_file_path
            - mode: search/training
            - sample_mode: random/exhaustive
  """
  config['mode'] = 'search'
  config['opt_latency'] = -1
  config['opt_resource'] = {}
  print(cmd)
  if config['hw_info']['URAM'] > 0:
    cmd += ' --uram'
  print(cmd)

  # Allocate the directory for training files
  sys_cmd = "rm -rf ./polysa.tmp/optimizer/search"
  if config['verbose']:
    print("[PolySA Optimizer] Execute CMD: " + sys_cmd)
  ret = subprocess.run(sys_cmd.split(), stdout=config['stdout'])
  sys_cmd = "mkdir ./polysa.tmp/optimizer/search"
  if config['verbose']:
    print("[PolySA Optimizer] Execute CMD: " + sys_cmd)
  ret = subprocess.run(sys_cmd.split(), stdout=config['stdout'])

  config['monitor'] = {'design_cnt': 0, \
                       'time': None, \
                       'boundary_design': {'latency': -1, 'dsp_underuse': 0}, \
                       'latency_pruned': 0, \
                       'res_pruned': {'FF': 0, 'LUT': 0, 'BRAM': 0, 'URAM': 0, 'DSP': 0}, \
                       'lat_est_time': 0, \
                       'res_est_time': 0, \
                       'lat_est_n': 0, \
                       'res_est_n': 0
                       }
  config['monitor']['time'] = time.time()

  # Random sample the design space
  if config['setting']['search']['random_init'] == 1:
    time1 = time.time()
    warmup_opt_latency = -1
    # In the warm-up seaching, we disable the lower bounds for tiling factors
    config['warmup_search'] = 1
    config['sample_mode'] = 'random'
    config['n_sample'] = config['setting']['search']['random_n_sample']
    n_try = 0
    while n_try < config['setting']['search']['random_time_out']:
      print('[PolySA Optimizer] Run random search to warm up...')
      explore_space_time(cmd, config)
      if warmup_opt_latency == -1:
        warmup_opt_latency = config['opt_latency']
      else:
        warmup_opt_latency = min(warmup_opt_latency, config['opt_latency'])
      print('[PolySA Optimizer] Warm-up opt. latency: %d' % (config['opt_latency']))
      n_try += 1
    time2 = time.time()
    elapsed_time = time2 - time1
    print('[PolySA Optimizer] Elapsed time(s): %d' % (elapsed_time))
    config['opt_latency'] = warmup_opt_latency
#    print('lat_est_time: ', config['monitor']['lat_est_time'])
#    print('res_est_time: ', config['monitor']['res_est_time'])

    config['monitor'] = {'design_cnt': 0, \
                         'time': None, \
                         'boundary_design': {'latency': -1, 'dsp_underuse': 0}, \
                         'latency_pruned': 0, \
                         'res_pruned': {'FF': 0, 'LUT': 0, 'BRAM': 0, 'URAM': 0, 'DSP': 0}, \
                         'lat_est_time': 0, \
                         'res_est_time': 0, \
                         'lat_est_n': 0, \
                         'res_est_n': 0
                         }
    config['monitor']['time'] = time.time()

  # Start the exploration with the space_time stage
  config['warmup_search'] = 0
  if config['setting']['search']['n_sample'] == -1:
    config['sample_mode'] = 'exhaustive'
  else:
    config['sample_mode'] = 'random'
  config['n_sample'] = config['setting']['search']['n_sample']
  time1 = time.time()
  explore_space_time(cmd, config)
  time2 = time.time()
  elapsed_time = time2 - time1
  print('[PolySA Optimizer] Elapsed time(s): %d' % (elapsed_time))

  # Print out the optimal design
  if config['setting']['search']['n_job'] == 1:
    print("[PolySA Optimizer] Total design count: " + str(config['monitor']['design_cnt']))
#    print("[PolySA Optimizer] Latency pruned design count: " + str(config['monitor']['latency_pruned_design_cnt']))
#    print("[PolySA Optimizer] Resource pruned design count: " + str(config['monitor']['resource_pruned_design_cnt']))
  print("[PolySA Optimizer] Optimal design latency: " + str(config['opt_latency']))
  print("[PolySA Optimizer] Optimal design resource: ")
  print(config['opt_resource'])

  # Select the top designs for frequency DSE
  print('[PolySA Optimizer] Extracting cycle DSE results...')
  if os.path.exists('./polysa.tmp/optimizer/search/cycle_dse_results'):
    sys_cmd = 'rm -rf ./polysa.tmp/optimizer/search/cycle_dse_results'
    if config['verbose']:
      print('[PolySA Optimizer] Execute CMD: ' + sys_cmd)
    ret = subprocess.run(sys_cmd.split(), stdout=config['stdout'])

  sys_cmd = 'mkdir ./polysa.tmp/optimizer/search/cycle_dse_results'
  if config['verbose']:
    print('[PolySA Optimizer] Execute CMD: ' + sys_cmd)
  ret = subprocess.run(sys_cmd.split(), stdout=config['stdout'])
  extract_cycle_dse_results(config)

def xilinx_run(cmd, info, setting, training, search, verbose):
  """ Design space exploration on Xilinx platform

  The following four stages are involved in the DSE:
  - space time transformation
  - array partitioning
  - latency hiding
  - simd vectorization

  The DSE include two phases: training phase and exploration phase.
  In the training phase, for each systolic array candidate, we will generate a set of tuning parameters for the
  later three stages. This step creates a suite of micro-benchmarks. We will use this benchmark to train regression models
  for latency and resource usage of the design.

  After the training stage is done, we enter the exploration phase. In this phase, for each systolic array candidate,
  we will explore all different tiling factors in the later three stages.
  After the tuning parameters of each stage is determined, we estimate the latency and resource usage of the design using
  the pre-trained regression model.

  At last, the design with the least latency and under the resource constraints is selected.

  Args:
    cmd: command line to run PolySA
    info: FPGA platform hardware resource information
    setting: optimizer settings
    training: enable training phase
    search: enable searching phase
    verbose: provide verbose information
  """

  # Initialize the global configuration
  config = {}
  config['setting'] = setting
  config['verbose'] = verbose
  if config['verbose']:
    config['stdout'] = None
  else:
    config['stdout'] = subprocess.DEVNULL
  config['work_dir'] = 'polysa.tmp'

  # Set two-level-buffer
  if cmd.find('two-level-buffer') != -1:
    config['two_level_buffer'] = 1
  else:
    config['two_level_buffer'] = 0

  # Load the resource information
  with open(info) as f:
    hw_info = json.load(f)
  config['hw_info'] = hw_info

  # Training phase
  # Generate the polysa_config file
  polysa_config = {"space_time": {"mode": "manual"}, \
                   "array_part": {"enable": 1, "mode": "manual"}, \
                   "array_part_L2": {"enable": config['two_level_buffer'], "mode": "manual"}, \
                   "latency": {"enable": 1, "mode": "manual"}, \
                   "simd": {"enable": 1, "mode": "manual"}}
  with open('polysa.tmp/polysa_config.json', 'w') as f:
    json.dump(polysa_config, f, indent=4)
  config['polysa_config'] = polysa_config

  # Load the SIMD info file
  kernel_file_path = cmd.split()[1]
  kernel_file_path = kernel_file_path.rsplit('/', 1)[0]
  config['kernel_file_path'] = kernel_file_path
  with open(kernel_file_path + '/simd_info.json', 'r') as f:
    simd_info = json.load(f)
  config['simd_info'] = simd_info

  # Clear the content
  cmd2 = 'rm ./polysa.tmp/latency_est/*'
  if config['verbose']:
    print('[PolySA Optimizer] Execute CMD: ' + cmd2)
  ret = subprocess.Popen(cmd2, shell=True, stdout=config['stdout'])

  cmd2 = 'rm ./polysa.tmp/resource_est/*'
  if config['verbose']:
    print('[PolySA Optimizer] Execute CMD: ' + cmd2)
  ret = subprocess.Popen(cmd2, shell=True, stdout=config['stdout'])

  cmd2 = 'rm ./polysa.tmp/src/*'
  if config['verbose']:
    print('[PolySA Optimizer] Execute CMD: ' + cmd2)
  ret = subprocess.Popen(cmd2, shell=True, stdout=config['stdout'])

  # Training phase
  if training:
    print('[PolySA Optimizer] Run training phase...')
    train_xilinx(cmd, config)

  # Search phase
  if search:
    print('[PolySA Optimizer] Run search phase...')
    search_xilinx(cmd, config)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='==== PolySA Optimizer ====')
  parser.add_argument('-c', '--cmd', metavar='CMD', required=True, help='PolySA command line')
  parser.add_argument('-i', '--info', metavar='INFO', required=True, help='hardware resource information')
  parser.add_argument('-s', '--setting', metavar='SETTING', required=False, default='polysa.config/optimizer_settings.json', help='optimizer settings')
  parser.add_argument('-p', '--platform', metavar='PLATFORM', required=True, help='hardware platform: intel/xilinx')
  parser.add_argument('--training', action='store_true', help='run training phase')
  parser.add_argument('--search', action='store_true', help='run search phase')
  parser.add_argument('--verbose', action='store_true', help='provide verbose information')

  args = parser.parse_args()

  # Parse the settings into a dict
  with open(args.setting) as f:
    setting = json.load(f)

  if args.platform == 'intel':
    print("Intel platform not supported yet!")
  elif args.platform == 'xilinx':
    xilinx_run(args.cmd, args.info, setting, args.training, args.search, args.verbose)
