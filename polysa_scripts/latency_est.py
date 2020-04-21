import sys
import argparse
import re
from os import listdir
from os import path
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
import pprint
import resource_est as res_model
import optimizer

def mean_absolute_percentage_error(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  error = np.divide((y_true - y_pred), y_true, out=np.zeros_like(-y_pred), where=y_true!=0)
  return np.mean(np.abs(error)) * 100

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
      if module.find('wrapper') != -1:
        continue

      if module not in module_list:
        module_list.append(module)
        stmt_list[module] = []

  for latency_info in latency_infos:
    loop_infos = latency_info['loop_infos']
    for module_name in module_list:
      if module_name.find('wrapper') != -1:
        continue

      if module_name not in loop_infos:
        continue
      loop_info = loop_infos[module_name]
      if module_name in latency_info['module_grouped']:
        module_group = latency_info['module_grouped'][module_name]
      else:
        module_group = None
      module_stmt_list = extract_module_stmt_list_xilinx(loop_info, module_group)
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

##debug
#  idx = 0
  for latency_info in latency_infos:
    loop_infos = latency_info['loop_infos']
#    print("************************** index", idx)
#    idx += 1
    for module in module_list:
      if module.find('wrapper') != -1:
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
##debug
#          print("****", module)
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

def parse_hls_rpt_loop(loop, stmt_info):
  """ Extract the loop details from the Xilinx HLS rpt

  Args:
    loop: loop item in the XML report
    stmt_info: the current item in the dict that points to the current loop item
  """

  loop_name = loop.tag
#  print(loop_name)
  stmt_info[loop_name] = {}
  # Parse the loop information
  for loop_child in loop:
    if loop_child.tag == 'TripCount':
      if loop_child.find('range') == None:
        if loop_child.text == 'undef':
          stmt_info[loop_name]['TripCount'] = -1
        else:
          stmt_info[loop_name]['TripCount'] = int(loop_child.text)
      else:
        # The trip count is in min/max range. We will only grab the max value
        stmt_info[loop_name]['TripCount'] = int(loop_child.find('range').find('max').text)
    elif loop_child.tag == 'Latency':
#      print(loop_child.text)
      if loop_child.find('range') == None:
        if loop_child.text == 'undef':
          stmt_info[loop_name]['Latency'] = -1
        else:
          stmt_info[loop_name]['Latency'] = int(loop_child.text)
      else:
        # The latency is in min/max range. We will only grab the max value
        stmt_info[loop_name]['Latency'] = int(loop_child.find('range').find('max').text)
#        print('here')
#        print(stmt_info[loop_name]['Latency'])
    elif loop_child.tag == 'IterationLatency':
      if loop_child.find('range') == None:
        if loop_child.text == 'undef':
          stmt_info[loop_name]['IterationLatency'] = -1
        else:
          stmt_info[loop_name]['IterationLatency'] = int(loop_child.text)
      else:
        stmt_info[loop_name]['Latency'] = int(loop_child.find('range').find('max').text)
    elif loop_child.tag == 'PipelineII':
      stmt_info[loop_name]['PipelineII'] = int(loop_child.text)
    elif loop_child.tag == 'PipelineDepth':
      stmt_info[loop_name]['PipelineDepth'] = int(loop_child.text)
    elif loop_child.tag.startswith(loop_name):
      parse_hls_rpt_loop(loop_child, stmt_info[loop_name])

def extract_stmt_info(hls_rpt):
  """ Extract the loop details from the Xilinx HLS rpt

  We will extract the loop details from the hls rpt in XML format.
  The extracted informaiton is stored and returned in a dict.


  Args:
    hls_rpt: root to the hls_rpt in XML format

  """
  stmt_info = {}
  root = hls_rpt
  for loop_summary in root.iter('SummaryOfLoopLatency'):
    # Travese the child iteratively
    for loop in loop_summary:
      loop_name = loop.tag
      parse_hls_rpt_loop(loop, stmt_info)

  return stmt_info

def is_loop_struct_leaf_empty(loop_struct):
  """ Examine if the leaf node of the loop struct is empty

  Args:
    loop_struct: loop structure in JSON format
  """

  if "loop" in loop_struct:
    child = loop_struct['loop']['child']
    if child == None:
      return 1
    else:
      return is_loop_struct_leaf_empty(child)
  elif "mark" in loop_struct:
    child = loop_struct['mark']['child']
    if child == None:
      return 1
    else:
      return is_loop_struct_leaf_empty(child)
  elif "user" in loop_struct:
    child = loop_struct['user']['user_expr']
    if child == None:
      return 1
    else:
      return 0
  elif "block" in loop_struct:
    children = loop_struct['block']['child']
    if children == None:
      return 1
    else:
      for child in children:
        is_empty = is_loop_struct_leaf_empty(child)
        if is_empty == 0:
          return 0
      return 1
  elif "if" in loop_struct:
    if_struct = loop_struct['if']
    then_block = if_struct['then']
    is_empty = is_loop_struct_leaf_empty(then_block)
    if is_empty == 0:
      return 0
    if 'else' in if_struct:
      else_block = if_struct['else']
      is_empty = is_loop_struct_leaf_empty(else_block)
      if is_empty == 0:
        return 0
      return 1

  return 1

def loop_struct_has_for_loop(loop_struct):
  """ Examine if the leaf node of the loop struct has any for loop

  Args:
    loop_struct: loop structure in JSON format
  """
  if "loop" in loop_struct:
    return 1
  elif "mark" in loop_struct:
    child = loop_struct['mark']['child']
    if child == None:
      return 0
    else:
      return loop_struct_has_for_loop(child)
  elif "user" in loop_struct:
    child = loop_struct['user']['user_expr']
    return 0
  elif "block" in loop_struct:
    children = loop_struct['block']['child']
    if children == None:
      return 0
    else:
      for child in children:
        has_for_loop = loop_struct_has_for_loop(child)
        if has_for_loop == 1:
          return 1
      return 0
  elif "if" in loop_struct:
    if_struct = loop_struct['if']
    then_block = if_struct['then']
    has_for_loop = loop_struct_has_for_loop(then_block)
    if has_for_loop == 1:
      return 1
    if 'else' in if_struct:
      else_block = if_struct['else']
      has_for_loop = loop_struct_has_for_loop(else_block)
      if has_for_loop == 1:
        return 1
    return 0

  return 0

## Buggy, do not use.
#def est_module_latency_xilinx(loop_struct, config):
#  """ Estimate the latency of the module on Xilinx platform
#
#  Args:
#    loop_struct: loop structure in JSON format
#    config: global dict containing the following structs:
#            - context: the context of outer parameters and iterators
#            - latency: the accumulative latency
#            - loop_prefix: the loop prefix at the current level
#            - loop_offset: the loop offset at the current level
#            - module_type: the module type 0 - default 1 - outer 2 - inter_trans 3 - intra_trans
#  """
#
#  latency = config['latency']
#  if "loop" in loop_struct:
#    # Extract the loop information
#    loop = loop_struct['loop']
#    loop_info = loop['loop_info']
#    lb = loop_info['lb']
#    ub = loop_info['ub']
#    iterator = loop_info['iter']
#    # check if lb/ub is number
#    if lb.isnumeric():
#      lb_n = int(lb)
#    else:
#      lb_n = 0
#      if config['verbose']:
#        print('[WARNING] Lower bound of iterator ' + iterator + ' is ' + lb + ', set as 0 by default.')
#    if ub.isnumeric():
#      ub_n = int(ub)
#    else:
#      # Check the outer parameters and iterators and compute the maximal value
#      print('[ERROR] Upper bound of iterator ' + iterator + ' is ' + ub + ', not supported yet!')
#      sys.exit()
#      # TODO: Use SymPy to substitute the symbols
#    config['context'][iterator] = {}
#    config['context'][iterator]['lb'] = lb_n
#    config['context'][iterator]['ub'] = ub_n
#    if config['under_unroll'] == 0:
#      latency = latency * (ub_n - lb_n + 1)
#      config['latency'] = latency
#
#    child = loop['child']
#    # if outer module, we will need to update loop_prefix at each loop level
#    if config['module_type'] == 1:
#      if config['loop_prefix'] == 'Loop':
#        config['loop_prefix'] = config['loop_prefix'] + str(config['loop_offset'])
#      else:
#        config['loop_prefix'] = config['loop_prefix'] + '.' + str(config['loop_offset'])
#      config['stmt_info'] = config['stmt_info'][config['loop_prefix']]
#
#    # Store the current for loop
#    config['last_for']['iter'] = iterator
#    config['last_for']['lb'] = lb_n
#    config['last_for']['ub'] = ub_n
#    if config['under_coalesce'] == 1:
#      config['last_for']['under_coalesce'] = 1
#    else:
#      config['last_for']['under_coalesce'] = 0
#    est_module_latency_xilinx(child, config)
#  elif "mark" in loop_struct:
#    mark = loop_struct['mark']
#    mark_name = mark['mark_name']
#    # If we meet the 'hls_unroll' mark, the loop below no longer counts in to the loop iteration
#    if mark_name == 'simd':
#      config['under_unroll'] = 1
#    if mark_name == 'access_coalesce':
#      config['under_coalesce'] = 1
#    child = mark['child']
#    est_module_latency_xilinx(child, config)
#  elif "user" in loop_struct:
#    user = loop_struct['user']
#    user_expr = user['user_expr']
#    config['under_unroll'] = 0
#    config['under_coalesce'] = 0
#    if config['module_type'] == 1:
#      # For outer module, we directly return
#      if config['latency'] == 1:
#        config['latency'] = 0
#      return
#
#    # Extract the loop II and depth
#    if config['loop_prefix'] == 'Loop':
#      loop_name = config['loop_prefix'] + str(config['loop_offset'])
#    else:
#      loop_name = config['loop_prefix'] + '.' + str(config['loop_offset'])
#    stmt_info = config['stmt_info'][loop_name]
#    II = stmt_info['PipelineII']
#    depth = stmt_info['PipelineDepth']
#
#    if user_expr.find('dram') != -1:
#      # Extract the array name
#      module_name = config['module_name']
#      array_name = module_name.split('_')[0]
#      array_info = config['array_info'][array_name]
#
#      if config['last_for']['under_coalesce'] == 1:
#        # This statement accesses the dram
#        burst_len = (config['last_for']['ub'] - config['last_for']['lb']) # in bytes
#        # The HBM latency is 200ns
#        dram_latency = 200 / config['cycle'] + burst_len + depth
#        latency = latency / burst_len * dram_latency
#      else:
#        latency = latency * (200 / config['cycle'] + depth)
#    else:
#      latency = (latency - 1) * II + depth
#    config['latency'] = latency
#  elif "block" in loop_struct:
#    block = loop_struct['block']
#    block_child = block['child']
#
#    # Check if only one child is valid and the rest only contain the empty leaf node.
#    # If so, continue from the non-empty leaf node w/o further action
#    n_child = 0
#    for child in block_child:
#      is_empty = is_loop_struct_leaf_empty(child)
#      if is_empty == 0:
#        n_child += 1
#        single_child = child
#
#    if n_child == 1:
#      est_module_latency_xilinx(single_child, config)
#      return
#
#    # Check if the current block contains "simd" mark.
#    # If so, continue from "simd" branch w/o any further action
#    simd_child = 0
#    for child in block_child:
#      if "mark" in child:
#        mark_name = child['mark']['mark_name']
#        if mark_name == 'simd':
#          child = child['mark']['child']
#          simd_child = 1
#          break
#    if simd_child == 1:
#      est_module_latency_xilinx(child, config)
#      return
#
#    # Proceed as normal
#    latency = config['latency']
#    offset = 1
#    block_latency = 0
#    if config['module_type'] != 1:
#      if config['loop_prefix'] == 'Loop':
#        config['loop_prefix'] = config['loop_prefix'] + str(config['loop_offset'])
#      else:
#        config['loop_prefix'] = config['loop_prefix'] + '.' + str(config['loop_offset'])
#      config['stmt_info'] = config['stmt_info'][config['loop_prefix']]
#    for child in block_child:
#      config['loop_offset'] = offset
#      has_for_loop = loop_struct_has_for_loop(child)
#      if has_for_loop:
#        config['latency'] = 1
#        est_module_latency_xilinx(child, config)
#        offset += 1
#        # The functions in the block are assumed to be executed sequentially.
#        # We will sum them up
#        block_latency += config['latency']
#    latency = latency * max(block_latency, 1)
#    config['latency'] = latency

def extract_module_stmts_xilinx(loop_struct, config):
  if 'loop' in loop_struct:
    config['under_loop'] = 1
    loop = loop_struct['loop']
    child = loop['child']
    # if outer module, we will need to update loop_prefix at each loop level
    if config['module_type'] == 1:
      if config['loop_prefix'] == 'Loop':
        config['loop_prefix'] = config['loop_prefix'] + str(config['loop_offset'])
      else:
        config['loop_prefix'] = config['loop_prefix'] + '.' + str(config['loop_offset'])
    extract_module_stmts_xilinx(child, config)
  elif 'mark' in loop_struct:
    mark = loop_struct['mark']
    child = mark['child']
    extract_module_stmts_xilinx(child, config)
  elif 'user' in loop_struct:
    user = loop_struct['user']
    user_expr = user['user_expr']
    if config['module_type'] == 1:
      # For outer module, we directly return
      return
    if config['loop_prefix'] == 'Loop':
      loop_name = config['loop_prefix'] + str(config['loop_offset'])
    else:
      loop_name = config['loop_prefix'] + '.' + str(config['loop_offset'])
    if user_expr.startswith('S_'):
      user_expr_prefix = user_expr.split('(')[0]
    else:
      user_expr_prefix = user_expr.split('.')[0]
    config['stmt_list'].append(config['module_name'] + '_' + loop_name + '_' + user_expr_prefix)
  elif 'block' in loop_struct:
    block = loop_struct['block']
    block_child = block['child']
    loop_prefix = config['loop_prefix']
    # Check if only one child is valid and the rest only contain the empty leaf node.
    # If so, continue from the non-empty leaf node w/o further action
    n_child = 0
    for child in block_child:
      is_empty = is_loop_struct_leaf_empty(child)
      if is_empty == 0:
        n_child += 1
        single_child = child
    if n_child == 1:
      extract_module_stmts_xilinx(single_child, config)
      return

    # Check if the current block contains 'simd' mark.
    # If so, continue from 'simd' branch w/o further action
    simd_child = 0
    for child in block_child:
      if 'mark' in child:
        mark_name = child['mark']['mark_name']
        if mark_name == 'simd':
          child = child['mark']['child']
          simd_child = 1
          break
    if simd_child == 1:
      extract_module_stmts_xilinx(child, config)
      return

    # Proceed as normal
    # Check if the child contains any non-simd loop. If yes, we will
    # update the loop prefix.
    for child in block_child:
      local_config = {}
      local_config['under_simd'] = 0
      has_non_simd_loop = loop_struct_has_non_simd_loop(child, local_config)
      if has_non_simd_loop:
        if config['module_type'] != 1 and config['under_loop'] == 1:
          if config['loop_prefix'] == 'Loop':
            config['loop_prefix'] = config['loop_prefix'] + str(config['loop_offset'])
          else:
            config['loop_prefix'] = config['loop_prefix'] + '.' + str(config['loop_offset'])
        break
    loop_prefix = config['loop_prefix']
    loop_offset = 1
    under_loop = config['under_loop']
    # If the block is under loop and all childrens are user nodes,
    # we will proceed and dive into the user nodes.
    all_user_child = 1
    for child in block_child:
      has_for_loop = loop_struct_has_for_loop(child)
      if has_for_loop:
        all_user_child = 0
        break
    for child in block_child:
      config['loop_offset'] = loop_offset
      config['loop_prefix'] = loop_prefix
      if under_loop == 1:
        config['under_loop'] = 0
      has_for_loop = loop_struct_has_for_loop(child)
      if all_user_child:
        extract_module_stmts_xilinx(child, config)
      else:
        if has_for_loop:
          extract_module_stmts_xilinx(child, config)
          loop_offset += 1
  elif 'if' in loop_struct:
    if_struct = loop_struct['if']
    then_block = if_struct['then']
    if config['module_type'] != 1 and config['under_loop'] == 1:
      if config['loop_prefix'] == 'Loop':
        config['loop_prefix'] = config['loop_prefix'] + str(config['loop_offset'])
      else:
        config['loop_prefix'] = config['loop_prefix'] + '.' + str(config['loop_offset'])
    loop_prefix = config['loop_prefix']
    loop_offset = config['loop_offset']
    has_for_loop = loop_struct_has_for_loop(then_block)
    if has_for_loop:
      extract_module_stmts_xilinx(then_block, config)
    if 'else' in if_struct:
      loop_offset += 1
      config['loop_offset'] = loop_offset
      else_block = if_struct['else']
      has_for_loop = loop_struct_has_for_loop(else_block)
      if has_for_loop:
        extract_module_stmts_xilinx(else_block, config)

def extract_module_stmt_list_xilinx(loop_info, module_group):
  """ Extract the module statement list from the loop info

  Args:
    loop_info: the loop information of the current module
    module_group: module grouping information
  """
  stmt_list = []
  module_name = loop_info['module_name']
  config = {}
  config['context'] = {}
  config['loop_prefix'] = 'Loop'
  config['loop_offset'] = 1
  config['module_name'] = loop_info['module_name']
  config['stmt_list'] = stmt_list
  # 0: default 1: outer 2: inter_trans 3: intra_trans
  config['module_type'] = 0
  if 'inter_trans' in module_name:
    config['module_type'] = 2
  elif 'intra_trans' in module_name:
    config['module_type'] = 3
  elif module_group != None:
    if 'inter_trans' in module_group or 'intra_trans' in module_group:
      config['module_type'] = 1
  config['under_loop'] = 0
  extract_module_stmts_xilinx(loop_info, config)

  return config['stmt_list']

def loop_struct_has_non_simd_loop(loop_struct, config):
  """ Examine if the leaf node of the loop struct has any non-simd loop

  Args:
    loop_struct: loop structure in JSON format
  """
  if "loop" in loop_struct:
    if config['under_simd'] == 1:
      return 0
    else:
      return 1
  elif "mark" in loop_struct:
    mark = loop_struct['mark']
    mark_name = mark['mark_name']
    if mark_name == 'simd':
      config['under_simd'] = 1
    child = mark['child']
    if child == None:
      return 0
    else:
      return loop_struct_has_non_simd_loop(child, config)
  elif "user" in loop_struct:
    return 0
  elif "block" in loop_struct:
    children = loop_struct['block']['child']
    if children == None:
      return 0
    else:
      for child in children:
        has_non_simd_loop = loop_struct_has_non_simd_loop(child, config)
        if has_non_simd_loop == 1:
          return 1
      return 0
  elif "if" in loop_struct:
    if_struct = loop_struct['if']
    then_block = if_struct['then']
    has_non_simd_loop = loop_struct_has_non_simd_loop(then_block, config)
    if has_non_simd_loop == 1:
      return 1
    if 'else' in if_struct:
      else_block = if_struct['else']
      has_non_simd_loop = loop_struct_has_non_simd_loop(else_block, config)
      if has_non_simd_loop == 1:
        return 1
    return 0

  return 0

def extract_module_stmt_latency(loop_struct, config):
  """ Extract the module staement latency on Xilinx platform

  Args:
    loop_strcut: loop structure in JSON format
    config: global dict containing the following structs:
            - context: the context of outer parameters and iterators
            - loop_prefix: the loop prefix at the current level
            - loop_offset: the loop offset at the current level
            - module_type: 0 - default 1 - outer 2 - inter_trans 3 - intra_trans
  """
  if 'loop' in loop_struct:
    config['under_loop'] = 1
    loop = loop_struct['loop']
    child = loop['child']
    # if outer module, we will need to update loop_prefix at each loop level
    if config['module_type'] == 1:
      if config['loop_prefix'] == 'Loop':
        config['loop_prefix'] = config['loop_prefix'] + str(config['loop_offset'])
      else:
        config['loop_prefix'] = config['loop_prefix'] + '.' + str(config['loop_offset'])
      config['stmt_info'] = config['stmt_info'][config['loop_prefix']]
    extract_module_stmt_latency(child, config)
  elif 'mark' in loop_struct:
    mark = loop_struct['mark']
    mark_name = mark['mark_name']
    child = mark['child']
    extract_module_stmt_latency(child, config)
  elif 'user' in loop_struct:
    user = loop_struct['user']
    user_expr = user['user_expr']
    if config['module_type'] == 1:
      # For outer module, we directly return
      return
    # Extract the loop II and depth
    if config['loop_prefix'] == 'Loop':
      loop_name = config['loop_prefix'] + str(config['loop_offset'])
    else:
      loop_name = config['loop_prefix'] + '.' + str(config['loop_offset'])
#    print(config['stmt_info'])
#    print(loop_name)
#    print(user_expr)
#    print(config['module_type'])
    if not bool(config['stmt_info']):
      # The stmt info is empty. The loop is fully unrolled by HLS.
      # Return NaN for both II and depth
      II = None
      depth = None
    else:
      stmt_info = config['stmt_info'][loop_name]
      if 'PipelineII' in stmt_info:
        II = stmt_info['PipelineII']
      else:
        II = None
      if 'PipelineDepth' in stmt_info:
        depth = stmt_info['PipelineDepth']
      else:
        depth = None
    if user_expr.startswith('S_'):
      user_expr_prefix = user_expr.split('(')[0]
    else:
      user_expr_prefix = user_expr.split('.')[0]
    stmt_name = config['module_name'] + '_' + loop_name + '_' + user_expr_prefix
    config['stmt_list'][stmt_name] = {'II': II, 'depth': depth}
  elif 'block' in loop_struct:
    block = loop_struct['block']
    block_child = block['child']
    # Check if only one child is valid and the rest only contain the empty leaf node.
    # If so, continue from the non-empty leaf node w/o further action
    n_child = 0
    for child in block_child:
      is_empty = is_loop_struct_leaf_empty(child)
      if is_empty == 0:
        n_child += 1
        single_child = child

    if n_child == 1:
      extract_module_stmt_latency(single_child, config)
      return
    # Check if the current block contains "simd" mark.
    # If so, continue from "simd" branch w/o any further action
    simd_child = 0
    for child in block_child:
      if "mark" in child:
        mark_name = child['mark']['mark_name']
        if mark_name == 'simd':
          child = child['mark']['child']
          simd_child = 1
          break
    if simd_child == 1:
      extract_module_stmt_latency(child, config)
      return
    # Proceed as normal
    # Check if the child contains any non-simd loop. If yes, we will
    # update the loop prefix.
    for child in block_child:
      local_config = {}
      local_config['under_simd'] = 0
      has_non_simd_loop = loop_struct_has_non_simd_loop(child, local_config)
      if has_non_simd_loop:
        if config['module_type'] != 1 and config['under_loop'] == 1:
          if config['loop_prefix'] == 'Loop':
            config['loop_prefix'] = config['loop_prefix'] + str(config['loop_offset'])
          else:
            config['loop_prefix'] = config['loop_prefix'] + '.' + str(config['loop_offset'])
          config['stmt_info'] = config['stmt_info'][config['loop_prefix']]
        break
    stmt_info = config['stmt_info']
    loop_prefix = config['loop_prefix']
    loop_offset = 1
    under_loop = config['under_loop']
    # If the block is under loop and all childrens are user nodes,
    # we will proceed and dive into the user nodes.
    all_user_child = 1
    for child in block_child:
      has_for_loop = loop_struct_has_for_loop(child)
      if has_for_loop:
        all_user_child = 0
        break
    for child in block_child:
      # Check if the child contains any for loop. If not, we will skip this child
      # w/o modifying the offset.
      config['loop_offset'] = loop_offset
      config['loop_prefix'] = loop_prefix
      config['stmt_info'] = stmt_info
#      print("block", loop_prefix, loop_offset)
      if under_loop == 1:
        config['under_loop'] = 0
      has_for_loop = loop_struct_has_for_loop(child)
      if all_user_child:
        extract_module_stmt_latency(child, config)
      else:
        if has_for_loop:
          extract_module_stmt_latency(child, config)
          loop_offset += 1
  elif 'if' in loop_struct:
    if_struct = loop_struct['if']
    then_block = if_struct['then']
    if config['module_type'] != 1 and config['under_loop'] == 1:
      if config['loop_prefix'] == 'Loop':
        config['loop_prefix'] = config['loop_prefix'] + str(config['loop_offset'])
      else:
        config['loop_prefix'] = config['loop_prefix'] + '.' + str(config['loop_offset'])
      config['stmt_info'] = config['stmt_info'][config['loop_prefix']]
      config['loop_offset'] = 1
    stmt_info = config['stmt_info']
    loop_prefix = config['loop_prefix']
    loop_offset = config['loop_offset']
#    print("if", loop_prefix)
    has_for_loop = loop_struct_has_for_loop(then_block)
    if has_for_loop:
      config['loop_prefix'] = loop_prefix
      config['loop_offset'] = loop_offset
      config['stmt_info'] = stmt_info
      extract_module_stmt_latency(then_block, config)
    if 'else' in if_struct:
      loop_offset += 1
      config['loop_prefix'] = loop_prefix
      config['loop_offset'] = loop_offset
      config['stmt_info'] = stmt_info
      else_block = if_struct['else']
      has_for_loop = loop_struct_has_for_loop(else_block)
      if has_for_loop:
        extract_module_stmt_latency(else_block, config)

def extract_module_stmt_latency_xilinx(loop_info, module_group, hls_rpt):
  """ Extract the module statment latency from the loop_info

  Args:
    loop_info: the loop information of the current module
    module_group: module grouping information
    hls_rpt: hls report
  """
#  print(module_group)
  stmt_list = {}
  module_name = loop_info['module_name']
  config = {}
  config['context'] = {}
  config['loop_prefix'] = 'Loop'
  config['loop_offset'] = 1
  config['module_name'] = module_name
  config['stmt_list'] = stmt_list
  # 0: default 1: outer 2: inter_trans 3: intra_trans
  config['module_type'] = 0
  if 'inter_trans' in module_name:
    config['module_type'] = 2
  elif 'intra_trans' in module_name:
    config['module_type'] = 3
  elif module_group != None:
    if 'inter_trans' in module_group or 'intra_trans' in module_group:
      config['module_type'] = 1
#  print(module_name)
#  print(ET.tostring(hls_rpt, encoding='utf8').decode('utf8'))
  stmt_info = extract_stmt_info(hls_rpt)
  config['stmt_info'] = stmt_info
#  print(stmt_info)
  config['under_loop'] = 0
  extract_module_stmt_latency(loop_info, config)

  return config['stmt_list']

def extract_latency_info(loop_info, hls_prj):
  """ Extract loop information of the design

  Returns a dictionary in the following format:
  - 'loop_infos': {
    - [module_name]: {
      'loop':
      ...
      }
    ...
    }
  - 'hls_rpts': {
    - [module_name]: {...}
    ...
    }
  - 'module_list': [...]
  - 'array_info': {...}
  - 'module_grouped': {...}

  Args:
    loop_info: directory that contains loop structure info
    hls_prj: directory contains hls project
  """
  loop_info_files = listdir(loop_info)
  loop_info_all = {}
  module_names = []
  # Load the loop info
  for f_name in loop_info_files:
    if f_name == 'array_info.json':
      with open(loop_info + '/' + f_name) as f:
        array_info = json.load(f)
    else:
      with open(loop_info + '/' + f_name) as f:
        loop_info_module = json.load(f)
        module_name = loop_info_module['module_name']
        loop_info_all[module_name] = loop_info_module
        module_names.append(module_name)

  hls_rpt_all = {}
  if hls_prj != None:
    # Load the HLS project
    hls_rpts = listdir(hls_prj + '/solution1/syn/report')
    hls_rpts = [hls_rpt for hls_rpt in hls_rpts if hls_rpt.endswith('.xml')]
    for f_name in hls_rpts:
      with open(hls_prj + '/solution1/syn/report/' + f_name) as f:
        if f_name.endswith('_csynth.xml'):
          tree = ET.parse(f)
          root = tree.getroot()
          module_name = f_name[:-11]
          while module_name[-1].isdigit():
            module_name = module_name[:-1]
          hls_rpt_all[module_name] = root

  module_grouped = {}
  # Place inter_trans and intra_trans module under the outer module
  for module_name in module_names:
    # intra_trans
    if module_name.find('intra_trans') != -1:
      module_name_prefix = module_name[:-12]
      if module_name_prefix not in module_grouped:
        module_grouped[module_name_prefix] = {}
      module_grouped[module_name_prefix]['intra_trans'] = module_name

      module_name_prefix = module_name_prefix + '_boundary'
      if module_name_prefix not in module_grouped:
        module_grouped[module_name_prefix] = {}
      module_grouped[module_name_prefix]['intra_trans'] = module_name
    # inter_trans
    elif module_name.find('inter_trans') != -1:
      if module_name.find('boundary') != -1:
        module_name_prefix = module_name[:-21] + '_boundary'
      else:
        module_name_prefix = module_name[:-12]

      if module_name_prefix not in module_grouped:
        module_grouped[module_name_prefix] = {}
      module_grouped[module_name_prefix]['inter_trans'] = module_name
    else:
      if module_name not in module_grouped:
        module_grouped[module_name] = {}

  latency_info = {'loop_infos': loop_info_all, 'hls_rpts': hls_rpt_all, 'module_list': module_names, 'module_grouped': module_grouped, 'array_info': array_info}

  return latency_info

def train_stmt_latency_models(kernel, df, module_list, stmt_list, verbose=0):
  """ Train the latency models for each module statment on Xilinx platforms

  Args:
    kernel: kernel_name
    df: dataframe that contains the design information
    module_list: module name list
    stmt_list: module statement list
    verbose: print verbose information
  """
  prj_dir = 'polysa.tmp/optimizer/training/' + kernel + '/latency_models'
  # Create the model folder
  sys_cmd = 'rm -rf ' + prj_dir
  print('[PolySA Optimizer] Execute CMD: ' + sys_cmd)
  ret = subprocess.run(sys_cmd.split())

  sys_cmd = 'mkdir ' + prj_dir
  print("[PolySA Optimizer] Execute CMD: " + sys_cmd)
  ret = subprocess.run(sys_cmd.split())

##debug
#  df.to_csv('debug_full.csv')

  # Split the train set and validate set
  feature_set = []
  pred_set  = []
  for module in module_list:
    if module.find('IO') != -1:
      feature_set.append(module + '_data_pack_inter')
      feature_set.append(module + '_data_pack_intra')
      feature_set.append(module + '_ele_size')
    else:
      feature_set.append(module + '_unroll')
  for module in stmt_list:
    for stmt in stmt_list[module]:
      pred_set.append(stmt + '_II')
      pred_set.append(stmt + '_depth')

  X = df.loc[:, feature_set]
  y = df.loc[:, pred_set]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

  II_mape = []
  depth_mape = []

  for module in module_list:
    module_feature_set = []
    if module.find('IO') != -1:
      module_feature_set.append(module + '_data_pack_inter')
      module_feature_set.append(module + '_data_pack_intra')
      module_feature_set.append(module + '_ele_size')
    else:
      module_feature_set.append(module + '_unroll')

    for stmt in stmt_list[module]:
      if verbose:
        print('[PolySA Optimizer] Validate latency model for stmt: ' + stmt)
      # II
      stmt_pred_set = [stmt + '_II']
      X_train_stmt = X_train.loc[:, module_feature_set]
      # Remove the missing value
#      X_train_stmt = X_train_stmt.dropna()
      y_train_stmt = y_train.loc[:, stmt_pred_set]

#      X_train_stmt.to_csv('debug_x.csv')
#      y_train_stmt.to_csv('debug_y.csv')

      y_train_stmt = y_train_stmt.dropna()
#      print(y_train_stmt.head())
#      print(y_train_stmt.index.values.tolist())
#      print(X_train_stmt.head())
#      print(X_train_stmt.iloc[0,:])
#debug

      if y_train_stmt.shape[0] == 0:
        # If there is no training smaple, we will set the default
        # values as 1
#        y_train_stmt = np.ones((X_train_stmt.shape[0], 1), dtype=float)
        X_train_stmt = X_train_stmt.dropna()
        y_train_stmt = np.ones((X_train_stmt.shape[0], 1), dtype=float)
      else:
        X_train_stmt = X_train_stmt.loc[y_train_stmt.index.values.tolist(), :]
#      print(X_train_stmt.head())
#      X_train_stmt.to_csv('debug_x.csv')
#      y_train_stmt.to_csv('debug_y.csv')

      regressor = LinearRegression()
      regressor.fit(X_train_stmt, y_train_stmt)
      model = regressor
      model_name = stmt + '_II_model'
      joblib_file = prj_dir + '/' + model_name + '.pkl'
      joblib.dump(model, joblib_file)
      # Validate the accuracy
      X_test_stmt = X_test.loc[:, module_feature_set]
      y_test_stmt = y_test.loc[:, stmt_pred_set]
      y_test_stmt = y_test_stmt.dropna()
      if y_test_stmt.shape[0] != 0:
        X_test_stmt = X_test_stmt.loc[y_test_stmt.index.values.tolist(), :]
        y_pred_stmt = model.predict(X_test_stmt)
#        print('II')
#        print(y_pred_stmt[:5])
        if verbose:
          print('\n======== II ========\n')
          print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test_stmt, y_pred_stmt))
          print('Mean Squared Error: ', metrics.mean_squared_error(y_test_stmt, y_pred_stmt))
          print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test_stmt, y_pred_stmt)))
          print('Mean Absolute Percentage Error: ', mean_absolute_percentage_error(y_test_stmt, y_pred_stmt))
        II_mape.append(mean_absolute_percentage_error(y_test_stmt, y_pred_stmt))

      # depth
      stmt_pred_set = [stmt + '_depth']
      X_train_stmt = X_train.loc[:, module_feature_set]
      # Remove the missing value
      y_train_stmt = y_train.loc[:, stmt_pred_set]
      y_train_stmt = y_train_stmt.dropna()
      if y_train_stmt.shape[0] == 0:
        # Use the default values as 1
        X_train_stmt = X_train_stmt.dropna()
        y_train_stmt = np.ones((X_train_stmt.shape[0], 1), dtype=float)
      else:
        X_train_stmt = X_train_stmt.loc[y_train_stmt.index.values.tolist(), :]

      regressor = LinearRegression()
      regressor.fit(X_train_stmt, y_train_stmt)
      model = regressor
      model_name = stmt + '_depth_model'
      joblib_file = prj_dir + '/' + model_name + '.pkl'
      joblib.dump(model, joblib_file)
      # Validate the accuracy
      X_test_stmt = X_test.loc[:, module_feature_set]
      y_test_stmt = y_test.loc[:, stmt_pred_set]
      y_test_stmt = y_test_stmt.dropna()
      if y_test_stmt.shape[0] != 0:
        X_test_stmt = X_test_stmt.loc[y_test_stmt.index.values.tolist(), :]
        y_pred_stmt = model.predict(X_test_stmt)
#        print('depth')
#        print(y_pred_stmt[:5])
        if verbose:
          print('\n======== Depth ========\n')
          print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test_stmt, y_pred_stmt))
          print('Mean Squared Error: ', metrics.mean_squared_error(y_test_stmt, y_pred_stmt))
          print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test_stmt, y_pred_stmt)))
          print('Mean Absolute Percentage Error: ', mean_absolute_percentage_error(y_test_stmt, y_pred_stmt))
        depth_mape.append(mean_absolute_percentage_error(y_test_stmt, y_pred_stmt))

  print('\n======== Module-Level Latency Model Validation Results ========\n')
  print('[PolySA Optimizer] II Mean Absolute Percentage Error (Geo. Mean): %.2f%%' % (mean(II_mape)))
  print('[PolySA Optimizer] Depth Mean Absolute Percentage Error (Geo. Mean): %.2f%%' % (mean(depth_mape)))

def predict_module_latency_xilinx(loop_struct, config):
  latency = config['latency']
  if "loop" in loop_struct:
    config['under_loop'] = 1
    # Extract the loop information
    loop = loop_struct['loop']
    loop_info = loop['loop_info']
    lb = loop_info['lb']
    ub = loop_info['ub']
    iterator = loop_info['iter']
    # check if lb/ub is number
    if lb.isnumeric():
      lb_n = int(lb)
    else:
      lb_n = 0
      if config['verbose']:
        print('[WARNING] Lower bound of iterator ' + iterator + ' is ' + lb + ', set as 0 by default.')
    if ub.isnumeric():
      ub_n = int(ub)
    else:
      # Check the outer parameters and iterators and compute the maximal value
      print('[ERROR] Upper bound of iterator ' + iterator + ' is ' + ub + ', not supported yet!')
      sys.exit()
      # TODO: Use SymPy to substitute the symbols
    config['context'][iterator] = {}
    config['context'][iterator]['lb'] = lb_n
    config['context'][iterator]['ub'] = ub_n
    if config['under_unroll'] == 0:
      latency = latency * (ub_n - lb_n + 1)
      config['latency'] = latency

    child = loop['child']
    # if outer module, we will need to update loop_prefix at each loop level
    if config['module_type'] == 1:
      if config['loop_prefix'] == 'Loop':
        config['loop_prefix'] = config['loop_prefix'] + str(config['loop_offset'])
      else:
        config['loop_prefix'] = config['loop_prefix'] + '.' + str(config['loop_offset'])

    # Store the current for loop
    config['last_for']['iter'] = iterator
    config['last_for']['lb'] = lb_n
    config['last_for']['ub'] = ub_n
    if config['under_coalesce'] == 1:
      config['last_for']['under_coalesce'] = 1
    else:
      config['last_for']['under_coalesce'] = 0
    predict_module_latency_xilinx(child, config)
  elif "mark" in loop_struct:
    mark = loop_struct['mark']
    mark_name = mark['mark_name']
    # If we meet the 'hls_unroll' mark, the loop below no longer counts in to the loop iteration
    if mark_name == 'simd':
      config['under_unroll'] = 1
    if mark_name == 'access_coalesce':
      config['under_coalesce'] = 1
    child = mark['child']
    predict_module_latency_xilinx(child, config)
  elif "user" in loop_struct:
    user = loop_struct['user']
    user_expr = user['user_expr']
    config['under_unroll'] = 0
    config['under_coalesce'] = 0
    if config['module_type'] == 1:
      # For outer module, we directly return
      if config['latency'] == 1:
        config['latency'] = 0
      return

    # Extract the loop II and depth
    if config['loop_prefix'] == 'Loop':
      loop_name = config['loop_prefix'] + str(config['loop_offset'])
    else:
      loop_name = config['loop_prefix'] + '.' + str(config['loop_offset'])
    # Load the latency model
    if user_expr.startswith('S_'):
      user_expr_prefix = user_expr.split('(')[0]
    else:
      user_expr_prefix = user_expr.split('.')[0]
    stmt_name = config['module_name'] + '_' + loop_name + '_' + user_expr_prefix
    II = max(np.asscalar(config['model_preds'][stmt_name + '_II']), 1)
#    II = 1
    depth = max(np.asscalar(config['model_preds'][stmt_name + '_depth']), 1)

    if user_expr.find('dram') != -1:
      # Extract the array name
      module_name = config['module_name']
      array_name = module_name.split('_')[0]
      array_info = config['array_info'][array_name]

      if config['last_for']['under_coalesce'] == 1:
        # This statement accesses the dram
        burst_len = (config['last_for']['ub'] - config['last_for']['lb']) # in bytes
        # The HBM latency is 200ns
        dram_latency = 200 / config['cycle'] + burst_len + depth
        latency = latency / burst_len * dram_latency
      else:
        latency = latency * (200 / config['cycle'] + depth)
    else:
      latency = (latency - 1) * II + depth
    config['latency'] = latency
  elif "block" in loop_struct:
    block = loop_struct['block']
    block_child = block['child']

    # Check if only one child is valid and the rest only contain the empty leaf node.
    # If so, continue from the non-empty leaf node w/o further action
    n_child = 0
    for child in block_child:
      is_empty = is_loop_struct_leaf_empty(child)
      if is_empty == 0:
        n_child += 1
        single_child = child

    if n_child == 1:
      predict_module_latency_xilinx(single_child, config)
      return

    # Check if the current block contains "simd" mark.
    # If so, continue from "simd" branch w/o any further action
    simd_child = 0
    for child in block_child:
      if "mark" in child:
        mark_name = child['mark']['mark_name']
        if mark_name == 'simd':
          config['under_unroll'] = 1
          child = child['mark']['child']
          simd_child = 1
          break
    if simd_child == 1:
      predict_module_latency_xilinx(child, config)
      return

    # Proceed as normal
    # Check if the child contains any non-simd loop. If yes, we will
    # update the loop prefix.
    for child in block_child:
      local_config = {}
      local_config['under_simd'] = 0
      has_non_simd_loop = loop_struct_has_non_simd_loop(child, local_config)
      if has_non_simd_loop:
        if config['module_type'] != 1 and config['under_loop'] == 1:
          if config['loop_prefix'] == 'Loop':
            config['loop_prefix'] = config['loop_prefix'] + str(config['loop_offset'])
          else:
            config['loop_prefix'] = config['loop_prefix'] + '.' + str(config['loop_offset'])
        break
    loop_prefix = config['loop_prefix']
    loop_offset = 1
    under_loop = config['under_loop']
    # If the block is under loop and all childrens are user nodes,
    # we will proceed and dive into the user nodes
    all_user_child = 1
    for child in block_child:
      has_for_loop = loop_struct_has_for_loop(child)
      if has_for_loop:
        all_user_child = 0
        break
    latency = config['latency']
    block_latency = 0
    for child in block_child:
      config['loop_offset'] = loop_offset
      config['loop_prefix'] = loop_prefix
      if under_loop == 1:
        config['under_loop'] = 0
      has_for_loop = loop_struct_has_for_loop(child)
      if all_user_child:
        # Select the statement with the longest latency
        config['latency'] = latency
        predict_module_latency_xilinx(child, config)
        block_latency = max(block_latency, config['latency'])
      else:
        if has_for_loop:
          config['latency'] = 1
          predict_module_latency_xilinx(child, config)
          loop_offset += 1
          block_latency += config['latency']
#          print(config['latency'])
#          print(block_latency)
    if all_user_child:
      latency = block_latency
    else:
      latency = latency * max(block_latency, 1)
    config['latency'] = latency
  elif 'if' in loop_struct:
    # For if then clause, we will treat it as similar as block by
    # adding up the latency of all sub blocks
    latency = config['latency']
    block_latency = 0
    if_struct = loop_struct['if']
    then_block = if_struct['then']
    if config['module_type'] != 1 and config['under_loop'] == 1:
      if config['loop_prefix'] == 'Loop':
        config['loop_prefix'] = config['loop_prefix'] + str(config['loop_offset'])
      else:
        config['loop_prefix'] = config['loop_prefix'] + '.' + str(config['loop_offset'])
    loop_prefix = config['loop_prefix']
    loop_offset = config['loop_offset']
    has_for_loop = loop_struct_has_for_loop(then_block)
    if has_for_loop:
      config['latency'] = 1
      predict_module_latency_xilinx(then_block, config)
      block_latency = max(block_latency, config['latency'])
    if 'else' in if_struct:
      loop_offset += 1
      config['loop_offset'] = loop_offset
      else_block = if_struct['else']
      has_for_loop = loop_struct_has_for_loop(else_block)
      if has_for_loop:
        config['latency'] = 1
        predict_module_latency_xilinx(else_block, config)
        block_latency = max(block_latency, config['latency'])
    latency = latency * max(block_latency, 1)
    config['latency'] = latency

def predict_kernel_latency(kernel, df, module_list, stmt_list, latency_info, verbose=0, cycle=5, early_stop=-1):
  # Load the latency models
  prj_dir = 'polysa.tmp/optimizer/training/' + kernel + '/latency_models'
  model_preds = {}
#  print(module_list)
#  print(stmt_list['PE'])
  for module in module_list:
#    print(module)
    module_feature_set = []
    if module.find('IO') != -1:
      module_feature_set.append(module + '_data_pack_inter')
      module_feature_set.append(module + '_data_pack_intra')
      module_feature_set.append(module + '_ele_size')
    else:
      module_feature_set.append(module + '_unroll')
    X = df.loc[:, module_feature_set]
#    print(X)

    for stmt in stmt_list[module]:
      model_name = stmt + '_II_model'
      joblib_file = prj_dir + '/' + model_name + '.pkl'
      if path.exists(joblib_file):
        model = joblib.load(joblib_file)
        II = model.predict(X)
      else:
        # Use the default value
        II = 1
      model_preds[stmt + '_II'] = np.round(II)

      model_name = stmt + '_depth_model'
      joblib_file = prj_dir + '/' + model_name + '.pkl'
      if path.exists(joblib_file):
        model = joblib.load(joblib_file)
        depth = model.predict(X)
      else:
        # Use the default value
        depth = 1
      model_preds[stmt + '_depth'] = np.round(depth)

#  pprint.pprint(model_preds)
  latency_all = {}
  config = {}
  config['model_preds'] = model_preds
  config['cycle'] = cycle # 200MHz by default
  config['verbose'] = verbose
  module_grouped = latency_info['module_grouped']
  array_info = latency_info['array_info']
  loop_infos = latency_info['loop_infos']
  for module_name in module_grouped:
    # TODO: temporary
    if 'dummy' in module_name:
      continue

    module = module_grouped[module_name]
    config['context'] = {}
    config['latency'] = 1
    config['loop_prefix'] = 'Loop'
    config['loop_offset'] = 1
    config['under_unroll'] = 0
    config['under_coalesce'] = 0
    config['under_loop'] = 0
    config['last_for'] = {}
    config['array_info'] = array_info
    config['module_name'] = module_name
    # 0: default 1: outer 2: inter_trans 3: intra_trans
    config['module_type'] = 0

    if 'inter_trans' in module or 'intra_trans' in module:
      # This is a filter module. We take it as double buffered by default.
      config['module_type'] = 1
      module_loop_info = loop_infos[module_name]
      predict_module_latency_xilinx(module_loop_info, config)
      outer_latency = config['latency']

      # inter module
      config['module_type'] = 2
      config['latency'] = 1
      config['loop_prefix'] = 'Loop'
      config['loop_offset'] = 1
      sub_module_name = module['inter_trans']
      config['module_name'] = sub_module_name
      module_loop_info = loop_infos[sub_module_name]
      predict_module_latency_xilinx(module_loop_info, config)
      inter_trans_latency = config['latency']

      # intra module
      config['module_type'] = 3
      config['latency'] = 1
      config['loop_prefix'] = 'Loop'
      config['loop_offset'] = 1
      sub_module_name = module['intra_trans']
      config['module_name'] = sub_module_name
      module_loop_info = loop_infos[sub_module_name]
      predict_module_latency_xilinx(module_loop_info, config)
      intra_trans_latency = config['latency']

      # This is not accurate. TODO: Consider the module type
      module_latency = outer_latency * (max(inter_trans_latency, intra_trans_latency)) + \
          max(inter_trans_latency, intra_trans_latency)
      latency_all[module_name] = module_latency
    else:
      module_loop_info = loop_infos[module_name]
      predict_module_latency_xilinx(module_loop_info, config)
      latency_all[module_name] = config['latency']

    # If we set early stop, we are using a baseline latency to compare.
    # If any of the module latency is greater than the baseline, we
    # will return immediately.
    if early_stop != -1:
      if config['latency'] > early_stop:
        return config['latency']

  pprint.pprint(latency_all)

  latency = 0
  for lat in latency_all:
    if latency_all[lat] > latency:
      latency = latency_all[lat]

  return latency

#def xilinx_run(loop_info, hls_prj, cycle, verbose):
#  """ Estimate the latency of the kernel on Xilinx platform
#
#  We will analyze the loop structure of each module in the kernel under the "loop_info" directory.
#  For the user statements, we will parse the hls report from the "hls_project" directory, and plug in the
#  latency extracted from the report.
#  The final latency is estimated as the maximal of all the module latencys.
#  latency = max(latency_1, latency_2, ...)
#
#  Note: This model is an approximation. Two major factors are omitted:
#  - DRAM contention. If the I/O module are mapped to the same I/O port, there will be DRAM contention which
#  increase the data loading latency. This model assumes that there is no DRAM contention.
#  - Critical path. The model uses the maximum of all the module latency. This assumes that all modules start
#  at the same time. This assumption overlooks the initial load-compute-store critial path. When there are
#  enough iterations, this critical path can be overlooked. In this model, this critical path latency is
#  not considered.
#
#  Args:
#    loop_info: directory contains loop structure info
#    hls_prj: directory contains hls project
#    cycle: cycle period of the design
#    verbose: flag to print verbose information
#  """
#
#  config = {}
#  config['verbose'] = verbose
#  config['cycle'] = cycle
#
#  module_names = []
#  # Load the loop info
#  loop_info_files = listdir(loop_info)
#  loop_info_all = {}
#  array_info = {}
#  for f_name in loop_info_files:
#    if f_name == 'array_info.json':
#      with open(loop_info + '/' + f_name) as f:
#        array_info = json.load(f)
#    else:
#      with open(loop_info + '/' + f_name) as f:
#        loop_info_module = json.load(f)
#        module_name = loop_info_module['module_name']
#        loop_info_all[module_name] = loop_info_module
#        module_names.append(module_name)
#
#  # Load the hls project
#  hls_rpts = listdir(hls_prj + '/prj/solution1/syn/report')
#  hls_rpts = [hls_rpt for hls_rpt in hls_rpts if hls_rpt.endswith('.xml')]
#  hls_rpt_all = {}
#  for f_name in hls_rpts:
#    with open(hls_prj + '/prj/solution1/syn/report/' + f_name) as f:
#      if f_name.endswith('_csynth.xml'):
#        tree = ET.parse(f)
#        # Get the root element
#        root = tree.getroot()
#        # Get the module name
#        # Get rid of the '_csynth.xml' suffix
#        module_name = f_name[:-11]
#        # For duplicate modules, get rid of the digits suffix.
#        # Only one report is kept.
#        while module_name[-1].isdigit():
#          module_name = module_name[:-1]
#        hls_rpt_all[module_name] = root
#
#  # For each hardware module in the loop_info_all, compute its latency
#  latency_all = {}
#  module_id = 0
#  # TODO: group modules by PREFIX
#  module_grouped = {}
#  # Place inter_trans and intra_trans module under the outer module
#  for module_name in module_names:
#    # intra_trans
#    if module_name.find('intra_trans') != -1:
#      module_name_prefix = module_name[:-12]
#      if module_name_prefix not in module_grouped:
#        module_grouped[module_grouped_prefix] = {}
#      module_grouped[module_name_prefix]['intra_trans'] = module_name
#
#      module_name_prefix = module_name_prefix + '_boundary'
#      if module_name_prefix not in module_grouped:
#        module_grouped[module_name_prefix] = {}
#      module_grouped[module_name_prefix]['intra_trans'] = module_name
#    # inter_trans
#    elif module_name.find('inter_trans') != -1:
#      if module_name.find('boundary') != -1:
#        module_name_prefix = module_name[:-21] + '_boundary'
#      else:
#        module_name_prefix = module_name[:-12]
#
#      if module_name_prefix not in module_grouped:
#        module_grouped[module_name_prefix] = {}
#      module_grouped[module_name_prefix]['inter_trans'] = module_name
#    else:
#      if module_name not in module_grouped:
#        module_grouped[module_name] = {}
#
##  print(module_grouped)
#  for module_name in module_grouped:
#    module = module_grouped[module_name]
#    config['context'] = {}
#    config['latency'] = 1
#    config['loop_prefix'] = 'Loop'
#    config['loop_offset'] = 1
#    config['under_unroll'] = 0
#    config['under_coalesce'] = 0
#    config['last_for'] = {}
#    config['array_info'] = array_info
#    config['module_name'] = module_name
#    # 0: default 1: outer 2: inter_trans 3: intra_trans
#    config['module_type'] = 0
#
##    if module_id != 12:
##      module_id += 1
##      continue
#
#    if 'inter_trans' in module or 'intra_trans' in module:
#      # This is a filter module. We take it as double buffered by default.
#      # TODO: fix it in non-db mode
#      config['module_type'] = 1
#      module_loop_info = loop_info_all[module_name]
#      print('[' + str(module_id) + '] Compute latency for module \"' + module_name + '\"')
#      # Fetch the hls report
#      module_hls_rpt = hls_rpt_all[module_name]
#      # Extract the stmt info
#      stmt_info = extract_stmt_info(module_hls_rpt)
#      config['stmt_info'] = stmt_info
#
#      est_module_latency_xilinx(module_loop_info, config)
#      outer_latency = config['latency']
#
#      # inter module
#      config['module_type'] = 2
#      config['latency'] = 1
#      config['loop_prefix'] = 'Loop'
#      config['loop_offset'] = 1
#      sub_module_name = module['inter_trans']
#      config['module_name'] = sub_module_name
##      print(sub_module_name)
#      module_loop_info = loop_info_all[sub_module_name]
#      # Fetch the hls report
#      module_hls_rpt = hls_rpt_all[sub_module_name]
#      # Extract the stmt info
#      stmt_info = extract_stmt_info(module_hls_rpt)
#      config['stmt_info'] = stmt_info
#
#      est_module_latency_xilinx(module_loop_info, config)
#      inter_trans_latency = config['latency']
#
#      # intra module
#      config['module_type'] = 3
#      config['latency'] = 1
#      config['loop_prefix'] = 'Loop'
#      config['loop_offset'] = 1
#      sub_module_name = module['intra_trans']
#      config['module_name'] = sub_module_name
#      module_loop_info = loop_info_all[sub_module_name]
#      # Fetch the hls report
#      module_hls_rpt = hls_rpt_all[sub_module_name]
#      # Extract the stmt info
#      stmt_info = extract_stmt_info(module_hls_rpt)
#      config['stmt_info'] = stmt_info
#
#      est_module_latency_xilinx(module_loop_info, config)
#      intra_trans_latency = config['latency']
#
#      # This is not accurate. TODO: Consider the module type
#      module_latency = outer_latency * (max(inter_trans_latency, intra_trans_latency)) + \
#        max(inter_trans_latency, intra_trans_latency)
#      print('[' + str(module_id) + '] Est. latency for module \"' + module_name + '\": ' + str(module_latency))
#      latency_all[module_name] = module_latency
#    else:
#      module_loop_info = loop_info_all[module_name]
#      print('[' + str(module_id) + '] Compute latency for module \"' + module_name + '\"')
#      # Fetch the hls report
#      module_hls_rpt = hls_rpt_all[module_name]
#      # Extract the stmt info
#      stmt_info = extract_stmt_info(module_hls_rpt)
##      print('[Debug] ' + str(stmt_info))
#      config['stmt_info'] = stmt_info
#
#      est_module_latency_xilinx(module_loop_info, config)
#      module_latency = config['latency']
#      print('[' + str(module_id) + '] Est. latency for module \"' + module_name + '\": ' + str(module_latency))
#      latency_all[module_name] = module_latency
#    module_id += 1
#
#  print(latency_all)
#
#  latency = 0
#  for lat in latency_all:
#    if latency_all[lat] > latency:
#      latency = latency_all[lat]
#
#  return latency

def xilinx_predict(design_dir, kernel, cycle, verbose):
  design_info = res_model.extract_design_info( \
      design_dir + '/resource_est/design_info.json', \
      design_dir + '/resource_est/design_info.dat',  \
      None)
  design_infos = [design_info]
  latency_info = extract_latency_info( \
      design_dir + '/latency_est', \
      None)
  latency_infos = [latency_info]
  ret = convert_latency_infos_to_df(latency_infos, design_infos)
  latency = predict_kernel_latency(kernel, ret['df'], ret['module_list'], \
      ret['stmt_list'], latency_info, verbose, int(cycle))
  return latency

#if __name__ == "__main__":
#  parser = argparse.ArgumentParser(description='==== PolySA Latency Estimator ====')
#  parser.add_argument('-i', '--loop-info', metavar='LOOP_INFO', required=True, help='directory of loop info')
#  parser.add_argument('-pr', '--hls-project', metavar='HLS_PROJECT', required=True, help='directory of HLS project')
#  parser.add_argument('-p', '--platform', metavar='PLATFORM', required=True, help='hardware platform: intel/xilinx')
#  parser.add_argument('-v', '--verbose', action='store_true', help='print verbose information')
#  parser.add_argument('-c', '--cycle', metavar='CYCLE', required=False, default=5, help='cycle period of the design (ns)')
#
#  args = parser.parse_args()
#
#  if args.platform == 'intel':
#    print("Intel platform not supported yet!")
#    latency = -1
#  elif args.platform == 'xilinx':
#    latency = xilinx_run(args.loop_info, args.hls_project, args.cycle, args.verbose)
#
#  print("Est. latency (cycle): " + str(latency))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='==== PolySA Latency Estimator ====')
  parser.add_argument('-d', '--design-dir', metavar='DESIGN_DIR', required=True, help='design directory')
  parser.add_argument('-k', '--kernel', metavar='KERNEL', required=True, help='kernel name')
  parser.add_argument('-p', '--platform', metavar='PLATFORM', required=True, help='hardware platform: intel/xilinx')
  parser.add_argument('-v', '--verbose', action='store_true', help='print verbose information')
  parser.add_argument('--train', action='store_true', help='training phase')
  parser.add_argument('--predict', action='store_true', help='predicting phase')
  parser.add_argument('-c', '--cycle', metavar='CYCLE', required=False, default=5, help='cycle period of the design (ns)')

  args = parser.parse_args()

  if args.platform == 'intel':
    print('Intel platform not supported yet!')
    latency = -1
  elif args.platform == 'xilinx':
#    latency = xilinx_run(args.design_dir, args.kernel, args.train, args.predict, args.cycle, args.verbose)
    latency = xilinx_predict(args.design_dir, args.kernel, args.cycle, args.verbose)

  print('Est. latency (cycle): ' + str(latency))
