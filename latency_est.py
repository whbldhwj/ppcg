import sys
import argparse
import re
from os import listdir
import json
import xml.etree.ElementTree as ET

def parse_hls_rpt_loop(loop, stmt_info):
  """ Extract the loop details from the Xilinx HLS rpt

  Args:
    loop: loop item in the XML report
    stmt_info: the current item in the dict that points to the current loop item
  """

  loop_name = loop.tag
  stmt_info[loop_name] = {}
  # Parse the loop information
  for loop_child in loop:
    if loop_child.tag == 'TripCount':
      stmt_info[loop_name]['TripCount'] = int(loop_child.text)
    elif loop_child.tag == 'Latency':
#      print(loop_child.text)
      if loop_child.find('range') == None:
        stmt_info[loop_name]['Latency'] = int(loop_child.text)
      else:
        # The latency is in min/max range. We will only grab the max value
        stmt_info[loop_name]['Latency'] = int(loop_child.find('range').find('max').text)
#        print('here')
#        print(stmt_info[loop_name]['Latency'])
    elif loop_child.tag == 'IterationLatency':
      if loop_child.find('range') == None:
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

  return 1

def est_module_latency_xilinx(loop_struct, config):
  """ Estimate the latency of the module on Xilinx platform

  Args:
    loop_struct: loop structure in JSON format
    config: global dict containing the following structs:
            - context: the context of outer parameters and iterators
            - latency: the accumulative latency
            - loop_prefix: the loop prefix at the current level
            - loop_offset: the loop offset at the current level
            - module_type: the module type 0 - default 1 - outer 2 - inter_trans 3 - intra_trans
  """

  latency = config['latency']
  if "loop" in loop_struct:
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
      config['stmt_info'] = config['stmt_info'][config['loop_prefix']]

    # Store the current for loop
    config['last_for']['iter'] = iterator
    config['last_for']['lb'] = lb_n
    config['last_for']['ub'] = ub_n
    if config['under_coalesce'] == 1:
      config['last_for']['under_coalesce'] = 1
    else:
      config['last_for']['under_coalesce'] = 0
    est_module_latency_xilinx(child, config)
  elif "mark" in loop_struct:
    mark = loop_struct['mark']
    mark_name = mark['mark_name']
    # If we meet the 'hls_unroll' mark, the loop below no longer counts in to the loop iteration
    if mark_name == 'hls_unroll':
      config['under_unroll'] = 1
    if mark_name == 'access_coalesce':
      config['under_coalesce'] = 1
    child = mark['child']
    est_module_latency_xilinx(child, config)
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
    stmt_info = config['stmt_info'][loop_name]
    II = stmt_info['PipelineII']
    depth = stmt_info['PipelineDepth']

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
      est_module_latency_xilinx(single_child, config)
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
      est_module_latency_xilinx(child, config)
      return

    # Proceed as normal
    latency = config['latency']
    offset = 1
    block_latency = 0
    if config['module_type'] != 1:
      if config['loop_prefix'] == 'Loop':
        config['loop_prefix'] = config['loop_prefix'] + str(config['loop_offset'])
      else:
        config['loop_prefix'] = config['loop_prefix'] + '.' + str(config['loop_offset'])
      config['stmt_info'] = config['stmt_info'][config['loop_prefix']]
    for child in block_child:
      config['loop_offset'] = offset
      config['latency'] = 1
      est_module_latency_xilinx(child, config)
      offset += 1
      # The functions in the block are assumed to be executed sequentially.
      # We will sum them up
      block_latency += config['latency']
    latency = latency * max(block_latency, 1)
    config['latency'] = latency

def xilinx_run(loop_info, hls_prj, cycle, verbose):
  """ Estimate the latency of the kernel on Xilinx platform

  We will analyze the loop structure of each module in the kernel under the "loop_info" directory.
  For the user statements, we will parse the hls report from the "hls_project" directory, and plug in the
  latency extracted from the report.
  The final latency is estimated as the maximal of all the module latencys.
  latency = max(latency_1, latency_2, ...)

  Note: This model is an approximation. Two major factors are omitted:
  - DRAM contention. If the I/O module are mapped to the same I/O port, there will be DRAM contention which
  increase the data loading latency. This model assumes that there is no DRAM contention.
  - Critical path. The model uses the maximum of all the module latency. This assumes that all modules start
  at the same time. This assumption overlooks the initial load-compute-store critial path. When there are
  enough iterations, this critical path can be overlooked. In this model, this critical path latency is
  not considered.

  Args:
    loop_info: directory contains loop structure info
    hls_prj: directory contains hls project
    cycle: cycle period of the design
    verbose: flag to print verbose information
  """

  config = {}
  config['verbose'] = verbose
  config['cycle'] = cycle

  module_names = []
  # Load the loop info
  loop_info_files = listdir(loop_info)
  loop_info_all = {}
  array_info = {}
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

  # For each hardware module in the loop_info_all, compute its latency
  latency_all = {}
  module_id = 0
  # TODO: group modules by PREFIX
  module_grouped = {}
  # Place inter_trans and intra_trans module under the outer module
  for module_name in module_names:
    # intra_trans
    if module_name.find('intra_trans') != -1:
      module_name_prefix = module_name[:-12]
      if module_name_prefix not in module_grouped:
        module_grouped[module_grouped_prefix] = {}
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

#  print(module_grouped)
  for module_name in module_grouped:
    module = module_grouped[module_name]
    config['context'] = {}
    config['latency'] = 1
    config['loop_prefix'] = 'Loop'
    config['loop_offset'] = 1
    config['under_unroll'] = 0
    config['under_coalesce'] = 0
    config['last_for'] = {}
    config['array_info'] = array_info
    config['module_name'] = module_name
    # 0: default 1: outer 2: inter_trans 3: intra_trans
    config['module_type'] = 0

#    if module_id != 12:
#      module_id += 1
#      continue

    if 'inter_trans' in module or 'intra_trans' in module:
      # This is a filter module. We take it as double buffered by default.
      # TODO: fix it in non-db mode
      config['module_type'] = 1
      module_loop_info = loop_info_all[module_name]
      print('[' + str(module_id) + '] Compute latency for module \"' + module_name + '\"')
      # Fetch the hls report
      module_hls_rpt = hls_rpt_all[module_name]
      # Extract the stmt info
      stmt_info = extract_stmt_info(module_hls_rpt)
      config['stmt_info'] = stmt_info

      est_module_latency_xilinx(module_loop_info, config)
      outer_latency = config['latency']

      # inter module
      config['module_type'] = 2
      config['latency'] = 1
      config['loop_prefix'] = 'Loop'
      config['loop_offset'] = 1
      sub_module_name = module['inter_trans']
      config['module_name'] = sub_module_name
#      print(sub_module_name)
      module_loop_info = loop_info_all[sub_module_name]
      # Fetch the hls report
      module_hls_rpt = hls_rpt_all[sub_module_name]
      # Extract the stmt info
      stmt_info = extract_stmt_info(module_hls_rpt)
      config['stmt_info'] = stmt_info

      est_module_latency_xilinx(module_loop_info, config)
      inter_trans_latency = config['latency']

      # intra module
      config['module_type'] = 3
      config['latency'] = 1
      config['loop_prefix'] = 'Loop'
      config['loop_offset'] = 1
      sub_module_name = module['intra_trans']
      config['module_name'] = sub_module_name
      module_loop_info = loop_info_all[sub_module_name]
      # Fetch the hls report
      module_hls_rpt = hls_rpt_all[sub_module_name]
      # Extract the stmt info
      stmt_info = extract_stmt_info(module_hls_rpt)
      config['stmt_info'] = stmt_info

      est_module_latency_xilinx(module_loop_info, config)
      intra_trans_latency = config['latency']

      # This is not accurate. TODO: Consider the module type
      module_latency = outer_latency * (max(inter_trans_latency, intra_trans_latency)) + \
        max(inter_trans_latency, intra_trans_latency)
      print('[' + str(module_id) + '] Est. latency for module \"' + module_name + '\": ' + str(module_latency))
      latency_all[module_name] = module_latency
    else:
      module_loop_info = loop_info_all[module_name]
      print('[' + str(module_id) + '] Compute latency for module \"' + module_name + '\"')
      # Fetch the hls report
      module_hls_rpt = hls_rpt_all[module_name]
      # Extract the stmt info
      stmt_info = extract_stmt_info(module_hls_rpt)
#      print('[Debug] ' + str(stmt_info))
      config['stmt_info'] = stmt_info

      est_module_latency_xilinx(module_loop_info, config)
      module_latency = config['latency']
      print('[' + str(module_id) + '] Est. latency for module \"' + module_name + '\": ' + str(module_latency))
      latency_all[module_name] = module_latency
    module_id += 1

  latency = 0
  for lat in latency_all:
    if latency_all[lat] > latency:
      latency = latency_all[lat]

  return latency

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='==== PolySA Latency Estimator ====')
  parser.add_argument('-i', '--loop-info', metavar='LOOP_INFO', required=True, help='directory of loop info')
  parser.add_argument('-pr', '--hls-project', metavar='HLS_PROJECT', required=True, help='directory of HLS project')
  parser.add_argument('-p', '--platform', metavar='PLATFORM', required=True, help='hardware platform: intel/xilinx')
  parser.add_argument('-v', '--verbose', action='store_true', help='print verbose information')
  parser.add_argument('-c', '--cycle', metavar='CYCLE', required=False, default=5, help='cycle period of the design (ns)')

  args = parser.parse_args()

  if args.platform == 'intel':
    print("Intel platform not supported yet!")
    latency = -1
  elif args.platform == 'xilinx':
    latency = xilinx_run(args.loop_info, args.hls_project, args.cycle, args.verbose)

  print("Est. latency (cycle): " + str(latency))
