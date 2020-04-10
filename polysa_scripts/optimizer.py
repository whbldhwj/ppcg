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

def generate_loop_candidates(loops, config, left, right):
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

  Args:
    loops: list containing the upper bounds of each loop
    config: configuration file
    left: indicates if lower bound should be included, i.e., 1
    right: indicates if upper bound should be included
  """
  if config['mode'] == 'training':
    setting = config['setting']['training']
  else:
    setting = config['setting']['search']
  # Sample each loop
  sample_list = []
  for loop in loops:
    index = int(np.floor(np.log2(loop)))
    if not right:
      ub = index - 1
    else:
      ub = index
    if not left:
      lb = 1
    else:
      lb = 0
    # Generate evenly spaced samples
    if setting['n_sample'] == -1:
      samples = range(lb, ub + 1)
    else:
      samples = np.linspace(lb, ub, setting['n_sample'])
      samples = set([int(s) for s in samples])
    samples = [np.power(2, int(s)) for s in samples]
    # double check
    if not left:
      if 1 in samples:
        samples.remove(1)
    if not right:
      if loop in samples:
        samples.remove(loop)
    sample_list.append(samples)
  # Generate cartesian product
#  print(sample_list)
  sample_loops = list(itertools.product(*sample_list))
  sample_loops = [list(tup) for tup in sample_loops]

  return sample_loops

def generate_simd_loop_candidates(loops, config):
  """ Generate candidate loops given the loop bounds
  This function samples each loop dimension given the sample number set in the config,
  then sets the rest dimensions as -1.
  Example: given the input loops as [4, 4], and n_sample as 2 set in config, it will first sample
  each loop dimension and generate two lists [2, 4], [2, 4]. Then, it fills the rest dimensions as -1
  and generates all candidates: [[2, -1], [4, -1], [-1, 2], [-1, 4]].

  Args:
    loops: list containing the upper bounds of each loop
    config: configuration file
  """
  if config['mode'] == 'training':
    setting = config['setting']['training']
  else:
    setting = config['setting']['search']
  # Sample each loop
  sample_list = []
  sample_loops = []
  for i in range(len(loops)):
    loop = loops[i]
    if loop == 1:
      samples = [1]
    else:
      index = int(np.floor(np.log2(loop)))
      lb = 0
      ub = index
      # Generate evenly spaced samples
      if setting['n_sample'] == -1:
        samples = range(lb, ub + 1)
      else:
        samples = np.linspace(lb, ub, setting['n_sample'])
        samples = set([int(s) for s in samples])
      samples = [np.power(2, int(s)) for s in samples]
    for sample in samples:
      sample_loop = [-1] * len(loops)
      sample_loop[i] = sample
      sample_loops.append(sample_loop)

  return sample_loops

def search_kernel(kernel_id, sa_sizes, config):
  """ Seach for the optimal design for the exploration process

  The following steps will be executed:
  1. Compile and run top_gen file to generate the top_kernel file.
  2. Predict the latency and resource of the design, select the optimal design by far.
  3. (Optional) Register the design into the database.

  Args:
    kernel_id: the selected kernel id from the space_time stage
    sa_sizes: the current sa_sizes configuration
    config: configuration file
  """
  # Compile top_gen file
  path = os.getcwd()
  cmd = 'g++ -o polysa.tmp/src/top_gen polysa.tmp/src/kernel_top_gen.cpp - I' + path + '/isl/include -L' + path + '/isl/.libs -lisl'
  print('Execute cmd: ' + cmd)
  ret = subprocess.run(cmd.split())
  if ret.returncode != 0:
    print('Cmd failed with error code: ' + str(ret.returncode))
  # Run the executable
  os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ':' + path + '/isl/.libs'
  cmd = './polysa.tmp/src/top_gen'
  print('Execute cmd: ' + cmd)
  ret = subprocess.run(cmd.split())
  if ret.returncode != 0:
    print('Cmd failed with error code: ' + str(ret.returncode))

  # Generate the final kernel file
  cmd = 'python3.6 ./polysa_scripts/codegen.py -c ./polysa.tmp/src/top.cpp -d ./polysa.tmp/src/kernel_kernel.cpp -p xilinx'
  print('Execute cmd: ' + cmd)
  ret = subprocess.run(cmd.split())
  if ret.returncode != 0:
    print('Cmd failed with error code: ' + str(ret.returncode))

  # Prepare the design files
  # Generate kernel folder
  if not os.path.exists('polysa.tmp/optimizer/search/kernel' + str(kernel_id)):
    cmd = 'mkdir polysa.tmp/optimizer/search/kernel' + str(kernel_id)
    print('Execute cmd: ' + cmd)
    ret = subprocess.run(cmd.split())

  # Generate sa_sizes annotation
  sa_sizes_cmd = generate_sa_sizes_cmd(sa_sizes)
  with open(folder_dir + '/sizes.annotation', 'w') as f:
    f.write(sa_sizes_cmd)

  # Cp files
  cmd = 'cp -r ./polysa.tmp/latency_est ' + folder_dir + '/'
  print('Execute cmd: ' + cmd)
  ret = subprocess.run(cmd.split())

  cmd = 'cp -r ./polysa.tmp/resource_est ' + folder_dir + '/'
  print('Execute cmd: ' + cmd)
  ret = subprocess.run(cmd.split())

  cmd = 'cp -r ./polysa.tmp/src ' + folder_dir + '/'
  print('Execute cmd: ' + cmd)
  ret = subprocess.run(cmd.split())

  cmd = 'cp ' + config['kernel_file_path'] + '/* ' + folder_dir + '/src/'
  print('Execute cmd: ' + cmd)
  ret = subprocess.Popen(cmd, shell=True)

  # Clear the content
  cmd = 'rm ./polysa.tmp/latency_est/*'
  print('Execute cmd: ' + cmd)
  ret = subprocess.Popen(cmd, shell=True)

  cmd = 'rm ./polysa.tmp/resource_est/*'
  print('Execute cmd: ' + cmd)
  ret = subprocess.Popen(cmd, shell=True)

  cmd = 'rm ./polysa.tmp/src/*'
  print('Execute cmd: ' + cmd)
  ret = subprocess.Popen(cmd, shell=True)

  return

def save_design_files(kernel_id, sa_sizes, config):
  path = os.getcwd()
  # Prepare the design files
  # Generate kernel folder
  if config['mode'] == 'training':
    if not os.path.exists('polysa.tmp/optimizer/training/kernel' + str(kernel_id)):
      cmd = 'mkdir polysa.tmp/optimizer/training/kernel' + str(kernel_id)
      print('[PolySA Optimizer] Execute CMD: ' + cmd)
      ret = subprocess.run(cmd.split())

    designs = os.listdir('polysa.tmp/optimizer/training/kernel' + str(kernel_id))
    design_id = len(designs)
    folder_dir = path + '/polysa.tmp/optimizer/training/kernel' + str(kernel_id) + '/design' + str(design_id)
    cmd = "mkdir " + folder_dir
    print("[PolySA Optimizer] Execute CMD: " + cmd)
    ret = subprocess.run(cmd.split())
  elif config['mode'] == 'search':
    if not os.path.exists('polysa.tmp/optimizer/search/kernel' + str(kernel_id)):
      cmd = 'mkdir polysa.tmp/optimizer/search/kernel' + str(kernel_id)
      print('[PolySA Optimizer] Execute CMD: ' + cmd)
      ret = subprocess.run(cmd.split())

    designs = os.listdir('polysa.tmp/optimizer/search/kernel' + str(kernel_id))
    design_id = len(designs)
    folder_dir = path + '/polysa.tmp/optimizer/search/kernel' + str(kernel_id) + '/design' + str(design_id)
    cmd = "mkdir " + folder_dir
    print("[PolySA Optimizer] Execute CMD: " + cmd)
    ret = subprocess.run(cmd.split())

  # Generate sa_sizes annotation
  sa_sizes_cmd = generate_sa_sizes_cmd(sa_sizes)
  with open(folder_dir + '/sizes.annotation', 'w') as f:
    f.write(sa_sizes_cmd)

  # Store latency and resource information
  if config['mode'] == 'search':
    with open(folder_dir + '/est.info', 'w') as f:
      f.write('latency est.: ' + str(config['opt_latency']) + '\n')
      f.write('resource est.: ' + str(config['opt_resource']) + '\n')

  # Cp files
  cmd = 'cp -r ./polysa.tmp/latency_est ' + folder_dir + '/'
  print('[PolySA Optimizer] Execute CMD: ' + cmd)
  ret = subprocess.run(cmd.split())

  cmd = 'cp -r ./polysa.tmp/resource_est ' + folder_dir + '/'
  print('[PolySA Optimizer] Execute CMD: ' + cmd)
  ret = subprocess.run(cmd.split())

  cmd = 'cp -r ./polysa.tmp/src ' + folder_dir + '/'
  print('[PolySA Optimizer] Execute CMD: ' + cmd)
  ret = subprocess.run(cmd.split())

  cmd = 'cp ' + config['kernel_file_path'] + '/* ' + folder_dir + '/src/'
  print('[PolySA Optimizer] Execute CMD: ' + cmd)
  ret = subprocess.Popen(cmd, shell=True)

def clear_design_files():
  cmd = 'rm ./polysa.tmp/latency_est/*'
  print('[PolySA Optimizer] Execute CMD: ' + cmd)
  ret = subprocess.Popen(cmd, shell=True)

  cmd = 'rm ./polysa.tmp/resource_est/*'
  print('[PolySA Optimizer] Execute CMD: ' + cmd)
  ret = subprocess.Popen(cmd, shell=True)

  cmd = 'rm ./polysa.tmp/src/*'
  print('[PolySA Optimizer] Execute CMD: ' + cmd)
  ret = subprocess.Popen(cmd, shell=True)

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
  cmd = 'g++ -o polysa.tmp/src/top_gen polysa.tmp/src/kernel_top_gen.cpp -I' + path + '/isl/include -L' + path + '/isl/.libs -lisl'
  print('[PolySA Optimizer] Execute CMD: ' + cmd)
  ret = subprocess.run(cmd.split())
  if ret.returncode != 0:
    print('[PolySA Optimizer] CMD failed with error code: ' + str(ret.returncode))
  # Run the executable
  os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ":" + path + '/isl/.libs'
  cmd = './polysa.tmp/src/top_gen'
  print('[PolySA Optimizer] Execute CMD: ' + cmd)
  ret = subprocess.run(cmd.split())
  if ret.returncode != 0:
    print('[PolySA Optimizer] CMD failed with error code: ' + str(ret.returncode))

  # Generate the final kernel file
  cmd = 'python3.6 ./polysa_scripts/codegen.py -c ./polysa.tmp/src/top.cpp -d ./polysa.tmp/src/kernel_kernel.cpp -p xilinx'
  print('[PolySA Optimizer] Execute CMD: ' + cmd)
  ret = subprocess.run(cmd.split())
  if ret.returncode != 0:
    print('[PolySA Optimizer] CMD failed with error code: ' + str(ret.returncode))

  if config['mode'] == 'training':
    save_design_files(kernel_id, sa_sizes, config)
  elif config['mode'] == 'search':
    # Predict latency and resource models
    design_info = res_model.extract_design_info(    \
        'polysa.tmp/resource_est/design_info.json', \
        'polysa.tmp/resource_est/design_info.dat',  \
        None)
    design_infos = [design_info]

    # Extract the latency infos
    latency_info = latency_model.extract_latency_info(  \
          'polysa.tmp/latency_est', \
          None)
    latency_infos = [latency_info]

    # Reorganize the loop_infos into a dataframe
    ret = convert_latency_infos_to_df(latency_infos, design_infos)
    # Predict latency
    latency = latency_model.predict_kernel_latency('kernel' + str(kernel_id), ret['df'], ret['module_list'], ret['stmt_list'], latency_info, config['verbose'], config['setting']['search']['cycle'])
    if config['opt_latency'] != -1 and latency > 2 * config['opt_latency']:
      clear_design_files()
      return

    # Reorganize the design_infos into a dataframe
    ret = convert_design_infos_to_df(design_infos)
    # Predict resource
    res_usage = res_model.predict_kernel_resource_usage('kernel' + str(kernel_id), ret['df'], ret['module_list'], ret['fifo_list'], design_info, config['verbose'])
    if res_usage['FF'] > config['hw_info']['FF']:
      clear_design_files()
      return
    if res_usage['LUT'] > config['hw_info']['LUT']:
      clear_design_files()
      return
    if res_usage['BRAM'] > config['hw_info']['BRAM']:
      clear_design_files()
      return
    if res_usage['DSP'] > config['hw_info']['DSP']:
      clear_design_files()
      return

    if config['opt_latency'] == -1 or latency < config['opt_latency']:
      config['opt_latency'] = latency
      config['opt_resource'] = res_usage
      # register the design
      save_design_files(kernel_id, sa_sizes, config)
#      print(config['opt_latency'])
#      print(config['opt_resource'])
#      sys.exit()

  clear_design_files()

def call_explore_kernel(kernel_id, sa_sizes, cmd_prefix, simd_info, config):
  sa_sizes_cmd = generate_sa_sizes_cmd(sa_sizes)
  # Execute the cmd
  print('[PolySA Optimizer] Execute CMD: ' + cmd_prefix + ' ' + sa_sizes_cmd)
  cmd = cmd_prefix.split() + [sa_sizes_cmd]
  proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,)
  for loop_i in simd_info:
    proc.communicate(str.encode(loop_i + '\n'))
  if proc.returncode != 0:
    print("[PolySA Optimizer] CMD failed with error code: " + str(proc.returncode))
#    # If the kernel is failed, we will skip this example and continue
#    return
  explore_kernel(kernel_id, sa_sizes, config)

def explore_simd(kernel_id, loops, cmd_prefix, sa_sizes, config):
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
  simd_loops_pool = generate_simd_loop_candidates(loops, config)
  if len(simd_loops_pool) == 0:
    # No available tiling options, we will disable this step and skip it
    config['polysa_config']['simd']['enable'] = 0
    with open('polysa.config/polysa_config.json', 'w') as f:
      json.dump(config['polysa_config'], f, indent=4)
    # Update sizes
    new_sa_sizes = sa_sizes.copy()
    # Call kernel
    call_explore_kernel(kernel_id, new_sa_sizes, cmd_prefix, config)
  else:
    for loop in simd_loops_pool:
      # Update sizes
      new_sa_sizes = sa_sizes.copy()
      new_sa_sizes.append('kernel[0]->simd' + str(loop))
      # call kernel
      call_explore_kernel(kernel_id, new_sa_sizes, cmd_prefix, simd_info, config)

def call_explore_simd(kernel_id, sa_sizes, cmd_prefix, simd_info, config):
  sa_sizes_cmd = generate_sa_sizes_cmd(sa_sizes)
  # Execute the cmd
  print("[PolySA Optimizer] Execute CMD: " + cmd_prefix + ' ' + sa_sizes_cmd)
  cmd = cmd_prefix.split() + [sa_sizes_cmd]
  proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,)
  for loop_i in simd_info:
    proc.communicate(str.encode(loop_i + '\n'))
  if proc.returncode != 0:
    print('[PolySA Optimizer] CMD failed with error code: ' + str(proc.returncode))
  # The program will terminate after the SIMD vectorization
  # Fetch the tuning info
  with open('polysa.tmp/tuning.json') as f:
    tuning = json.load(f)
  loops = tuning['simd']['tilable_loops']
  explore_simd(kernel_id, loops, cmd_prefix, sa_sizes, config)

def explore_latency_hiding(kernel_id, loops, cmd_prefix, sa_sizes, config):
  """ Generate latency hiding training candidates and proceed to simd vectorization

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
  latency_loops_pool = generate_loop_candidates(loops, config, 1, 0)
  if len(latency_loops_pool) == 0:
    # No available tiling options, we will disable this step and skip it
    config['polysa_config']['latency']['enable'] = 0
    with open('polysa.config/polysa_config.json', 'w') as f:
      json.dump(config['polysa_config'], f, indent=4)
    # Update the sizes
    new_sa_sizes = sa_sizes.copy()
    # Call simd
    call_explore_simd(kernel_id, new_sa_sizes, cmd_prefix, config)
  else:
    for loop in latency_loops_pool:
 #  for loop in [latency_loops_pool[2]]:
      # Update the sizes
      new_sa_sizes = sa_sizes.copy()
      new_sa_sizes.append('kernel[0]->latency' + str(loop))
      # Call simd
      call_explore_simd(kernel_id, new_sa_sizes, cmd_prefix, simd_info, config)

def call_explore_latency(kernel_id, sa_sizes, cmd_prefix, config):
  sa_sizes_cmd = generate_sa_sizes_cmd(sa_sizes)
  # Execute the cmd
  print("[PolySA Optimizer] Execute CMD: " + cmd_prefix + ' ' + sa_sizes_cmd)
  cmd = cmd_prefix.split() + [sa_sizes_cmd]
  ret = subprocess.run(cmd)
  if ret.returncode != 0:
    print("[PolySA Optimizer] CMD failed with error code: " + str(ret.returncode))
  # The program will terminate after the latency hiding
  # Fetch the tuning info
  with open ('polysa.tmp/tuning.json') as f:
    tuning = json.load(f)
  loops = tuning['latency']['tilable_loops']
  explore_latency_hiding(kernel_id, loops, cmd_prefix, sa_sizes, config)

def explore_array_part_L2(kernel_id, loops, cmd_prefix, sa_sizes, config):
  array_part_L2_loops_pool = generate_loop_candidates(loops, config, 1, 1)
  if len(array_part_L2_loops_pool) == 0:
    # No available tiling optioons, we will disable this step and skip it
    config['polysa_config']['array_part_L2']['enable'] = 0
    with open('polysa.config/polysa_config.json', 'w') as f:
      json.dump(config['polysa_config'], f, indent=4)
    # Update the sizes
    new_sa_sizes = sa_sizes.copy()
    # Call latency
    call_explore_latency(kernel_id, new_sa_sizes, cmd_prefix, config)
  else:
    for loop in array_part_L2_loops_pool:
      # Update the sizes
      new_sa_sizes = sa_sizes.copy()
      new_sa_sizes.append('kernel[0]->array_part_L2' + str(loop))
      # Call simd
      call_explore_latency(kernel_id, new_sa_sizes, cmd_prefix, config)

def call_explore_array_part_L2(kernel_id, sa_sizes, cmd_prefix, config):
  sa_sizes_cmd = generate_sa_sizes_cmd(sa_sizes)
  # Execute the cmd
  print("[PolySA Optimizer] Execute CMD: " + cmd_prefix + ' ' + sa_sizes_cmd)
  cmd = cmd_prefix.split() + [sa_sizes_cmd]
  ret = subprocess.run(cmd)
  if ret.returncode != 0:
    print("[PolySA Optimizer] CMD failed with error code: " + str(ret.returncode))
  # The program will terminate after the L2 array partitioning
  # Fetch the tuning info
  with open('polysa.tmp/tuning.json') as f:
    tuning = json.load(f)
  loops = tuning['array_part_L2']['tilable_loops']
  explore_array_part_L2(kernel_id, loops, cmd_prefix, sa_sizes, config)

def explore_array_part(kernel_id, cmd_prefix, config):
  """ Generate array partitioning training candidates and proceed to latency hiding

  Args:
    kernel_id: the selected kernel_id from the space_time stage
    cmd_prefix: the current cmd prefix
    config: tuning configuration
  """
  # Update the cmd
  sa_sizes = ['kernel[0]->space_time[' + str(kernel_id) + ']']
  sa_sizes_cmd = generate_sa_sizes_cmd(sa_sizes)
  cmd = cmd_prefix + ' ' + sa_sizes_cmd
  # Execute the cmd
  print("[PolySA Optimizer] Execute CMD: " + cmd)
  ret = subprocess.run(cmd.split())
  if ret.returncode != 0:
    print("[PolySA Optimizer] CMD failed with error code: " + str(ret.returncode))
  # The program will terminate after the array partitioning
  # Fetch the tuning info
  with open('polysa.tmp/tuning.json') as f:
    tuning = json.load(f)
  loops = tuning['array_part']['tilable_loops']
  # Generate a set of uniformly distributed tiling factors to proceed
  array_part_loops_pool = generate_loop_candidates(loops, config, 0, 1)
  if len(array_part_loops_pool) == 0:
    # No available tiling options, we will disable the step and skip it.
    # At the same time, two_level_buffer is disabled
    config['polysa_config']['array_part']['enable'] = 0
    config['polysa_config']['array_part_L2']['enable'] = 0
    with open('polysa.config/polysa_config.json', 'w') as f:
      json.dump(config['polysa_config'], f, indent=4)
    # Update the sizes
    new_sa_sizes = sa_sizes.copy()
    # Call latency
    call_explore_latency(kernel_id, new_sa_sizes, cmd_prefix, config)
  else:
    for loop in array_part_loops_pool:
#   for loop in [array_part_loops_pool[13]]:
      # Update the sizes
      new_sa_sizes = sa_sizes.copy()
      new_sa_sizes.append('kernel[0]->array_part' + str(loop))
      if config['two_level_buffer'] == 1:
        # Call array_part_L2
        call_explore_array_part_L2(kernel_id, new_sa_sizes, cmd_prefix, config)
      else:
        # Call latency
        call_explore_latency(kernel_id, new_sa_sizes, cmd_prefix, config)

def synth_train_samples(config):
#  # Set up the environment
#  cmd = 'source /opt/tools/xilinx/Vitis/2019.2/settings64.sh'
#  print('Execute cmd: ' + cmd)
#  ret = subprocess.run(cmd)

  # Copy the script.tcl to each training folder and execute the program
  kernels = os.listdir('polysa.tmp/optimizer/training')
  kernels = sorted(kernels, key = lambda x:int(x[6:]))
  for kernel in kernels:
    designs = os.listdir('polysa.tmp/optimizer/training/' + kernel)
    designs = sorted(designs, key = lambda x:int(x[6:]))
    for design in designs:
      prj_dir = 'polysa.tmp/optimizer/training/' + kernel + '/' + design
      # cp tcl to prj folder
      cmd = 'cp polysa_scripts/script.tcl ' + prj_dir + '/'
      print('[PolySA Optimizer] Execute CMD: ' + cmd)
      ret = subprocess.run(cmd.split())

      # Execute the tcl
      cwd = os.getcwd()
      os.chdir(prj_dir)

      cmd = 'vivado_hls -f script.tcl'
      print('[PolySA Optimizer] Execute cmd: ' + cmd)
      ret = subprocess.run(cmd.split())

      os.chdir(cwd)

def convert_latency_infos_to_df(latency_infos, design_infos):
  """ Convert the latency info into a dataframe

  For each design, we will scan the loop info of each module, for each statement
  to be estimated, we will extract the number from the HLS report and add it into the dataframe.
  The input latency_infos is a dictionary.
  Returns a dictionary, containing:
  - df: the dataframe that contains the design information
  - module_list: all the module names
  - stmt_list: all the module statements

  Args:
    latency_infos: list containing all loop infos
    design_infos: list containinig all design infos
  """
  module_list = []
  stmt_list = {}
  # Extract module_list and stmt_list
  for design_info in design_infos:
    modules = design_info['modules']
    for module in modules:
      # TODO: temporary, delete it later
      if module.find('dummy') != -1:
        continue
      if module not in module_list:
        module_list.append(module)
        stmt_list[module] = []

  for latency_info in latency_infos:
    loop_infos = latency_info['loop_infos']
    for module_name in module_list:
      # TODO: temporary, delete it later
      if module_name.find('dummy') != -1:
        continue

      loop_info = loop_infos[module_name]
      if module_name in latency_info['module_grouped']:
        module_group = latency_info['module_grouped'][module_name]
      else:
        module_group = None
      module_stmt_list = latency_model.extract_module_stmt_list_xilinx(loop_info, module_group)
      for module_stmt in module_stmt_list:
        if module_stmt not in stmt_list[module_name]:
          stmt_list[module_name].append(module_stmt)

  info_dict = {}
  # Initalization
  for module in module_list:
    if module.find('IO') != -1:
      # IO module
      info_dict[module + '_data_pack_inter'] = []
      info_dict[module + '_data_pack_intra'] = []
      info_dict[module + '_ele_size'] = []
    else:
      # PE module
      info_dict[module + '_unroll'] = []

    for stmt in stmt_list[module]:
      info_dict[stmt + '_II'] = []
      info_dict[stmt + '_depth'] = []

  for design_info in design_infos:
    modules = design_info['modules']
    for module in module_list:
      if module.find('IO') != -1:
        # IO module
        if module in modules:
          info_dict[module + '_data_pack_inter'].append(modules[module]['data_pack_inter'])
          info_dict[module + '_data_pack_intra'].append(modules[module]['data_pack_intra'])
          info_dict[module + '_ele_size'].append(modules[module]['ele_size'])
        else:
          info_dict[module + '_data_pack_inter'].append(None)
          info_dict[module + '_data_pack_intra'].append(None)
          info_dict[module + '_ele_size'].append(None)
      else:
        # PE module
        if module in modules:
          info_dict[module + '_unroll'].append(modules[module]['unroll'])
        else:
          info_dict[module + '_unroll'].append(None)

  for latency_info in latency_infos:
    loop_infos = latency_info['loop_infos']
    for module in module_list:
      # TODO: delete it later
      if module.find('dummy') != -1:
        continue

      if module in loop_infos:
        loop_info = loop_infos[module]
        if module in latency_info['module_grouped']:
          module_group = latency_info['module_grouped'][module]
        else:
          module_group = None
        if not bool(latency_info['hls_rpts']):
          for stmt in stmt_list[module]:
            info_dict[stmt + '_II'].append(None)
            info_dict[stmt + '_depth'].append(None)
        else:
          hls_rpt = latency_info['hls_rpts'][module]
          module_stmt_latency = latency_model.extract_module_stmt_latency_xilinx(loop_info, module_group, hls_rpt)
          for stmt in stmt_list[module]:
            if stmt in module_stmt_latency:
              info_dict[stmt + '_II'].append(module_stmt_latency[stmt]['II'])
              info_dict[stmt + '_depth'].append(module_stmt_latency[stmt]['depth'])
            else:
              info_dict[stmt + '_II'].append(None)
              info_dict[stmt + '_depth'].append(None)
      else:
        for stmt in stmt_list[module]:
          info_dict[stmt + '_II'].append(None)
          info_dict[stmt + '_depth'].append(None)

  df = pd.DataFrame(info_dict)
  return {'df': df, 'module_list': module_list, 'stmt_list': stmt_list}

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
      # TODO: temporary, delete it later
      if module.find('dummy') != -1:
        continue

      if module not in module_list:
        module_list.append(module)

  # Reorganize the design infos
  # Initialization
  info_dict['FF'] = []
  info_dict['LUT'] = []
  info_dict['DSP'] = []
  info_dict['BRAM'] = []
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
    info_dict[module + '_DSP'] = []

  for design_info in design_infos:
    # FF, LUT, BRAM, DSP
    info_dict['FF'].append(design_info['FF'])
    info_dict['LUT'].append(design_info['LUT'])
    info_dict['DSP'].append(design_info['DSP'])
    info_dict['BRAM'].append(design_info['BRAM'])

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
        info_dict[module + '_DSP'].append(modules[module]['DSP'])
      else:
        info_dict[module + '_FF'].append(None)
        info_dict[module + '_LUT'].append(None)
        info_dict[module + '_BRAM'].append(None)
        info_dict[module + '_DSP'].append(None)

#  print(info_dict['C_IO_L3_in_BRAM'][26])
  # Generate the dataframe
  df = pd.DataFrame(info_dict)
  return {'df': df, 'module_list': module_list, 'fifo_list': fifo_list}

def train_resource_models_xilinx(config):
  """ Train the resource models for Xilinx platforms

  Args:
    config: global parameters
  """
  kernels = os.listdir('polysa.tmp/optimizer/training')
  kernels = sorted(kernels)
  for kernel in kernels:
    print('[PolySA Optimizer] Train resource models for ' + kernel)
    designs = os.listdir('polysa.tmp/optimizer/training/' + kernel)
    if 'resource_models' in designs:
      designs.remove('resource_models')
    if 'latency_models' in designs:
      designs.remove('latency_models')
    designs = sorted(designs)
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
    ret = convert_design_infos_to_df(design_infos)
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
    ret = convert_latency_infos_to_df(latency_infos, design_infos)
    # Train for each statement and store the trained model
    latency_model.train_stmt_latency_models(kernel, ret['df'], ret['module_list'], ret['stmt_list'], config['verbose'])

def explore_space_time(cmd, config):
  """ Explore the stage of space time transformation

  Args:
    cmd: input user command
    config: global configuration
  """
  # Execute the cmd
  cmd_prefix = cmd + ' --config=polysa.config/polysa_config.json'
  print("[PolySA Optimizer] Execute CMD: " + cmd_prefix)
  ret = subprocess.run(cmd_prefix.split())
  if ret.returncode != 0:
    print("[PolySA Optimizer] CMD failed with error code: " + str(ret.returncode))
  # The program will terminate after the space-time transformation.
  # Fetch the tuning info
  with open('polysa.tmp/tuning.json') as f:
    tuning = json.load(f)
  n_kernel = tuning['space_time']['n_kernel']

  # Iterate through different kernels
  # TODO: temporarily commented out for debugging
#  for kernel_id in range(n_kernel):
  for kernel_id in range(3, 4):
    explore_array_part(kernel_id, cmd_prefix, config)

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

#  # Allocate the directory for training files
#  sys_cmd = "rm -rf ./polysa.tmp/optimizer/training"
#  print("[PolySA Optimizer] Execute CMD: " + sys_cmd)
#  ret = subprocess.run(sys_cmd.split())
#  sys_cmd = "mkdir ./polysa.tmp/optimizer/training"
#  print("[PolySA Optimizer] Execute CMD: " + sys_cmd)
#  ret = subprocess.run(sys_cmd.split())
#
#  # Start the exploration with the space_time stage
#  explore_space_time(cmd, config)
#
#  # Execute the HLS program to synthesize all the program
#  synth_train_samples(config)

#  # Train the linear regression models for FF, LUT, and DSPs
#  # For BRAM, we use the static analysis
#  train_resource_models_xilinx(config)
  # Train latency models
  train_latency_models_xilinx(config)

def search_xilinx(cmd, config):
  """ Design space exploration on Xilinx platforms

  Args:
    cmd: user cmd
    config: global configuration
  """
  config['mode'] = 'search'
  config['opt_latency'] = -1

  # Allocate the directory for training files
  sys_cmd = "rm -rf ./polysa.tmp/optimizer/search"
  print("[PolySA Optimizer] Execute CMD: " + sys_cmd)
  ret = subprocess.run(sys_cmd.split())
  sys_cmd = "mkdir ./polysa.tmp/optimizer/search"
  print("[PolySA Optimizer] Execute CMD: " + sys_cmd)
  ret = subprocess.run(sys_cmd.split())

  # Start the exploration with the space_time stage
  explore_space_time(cmd, config)

  # Print out the optimal design
  print("[PolySA Optimizer] Optimal design latency: " + str(config['opt_latency']))
  print("[PolySA Optimizer] Optimal design resource: ")
  print(config['opt_resource'])

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
  with open('polysa.config/polysa_config.json', 'w') as f:
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
  print('[PolySA Optimizer] Execute CMD: ' + cmd2)
  ret = subprocess.Popen(cmd2, shell=True)

  cmd2 = 'rm ./polysa.tmp/resource_est/*'
  print('[PolySA Optimizer] Execute CMD: ' + cmd2)
  ret = subprocess.Popen(cmd2, shell=True)

  cmd2 = 'rm ./polysa.tmp/src/*'
  print('[PolySA Optimizer] Execute CMD: ' + cmd2)
  ret = subprocess.Popen(cmd2, shell=True)

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
