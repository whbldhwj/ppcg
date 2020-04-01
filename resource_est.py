import sys
import argparse
import re
from os import listdir
import json
import xml.etree.ElementTree as ET
import numpy as np

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
    BRAM = BRAM_TDP_predict_xilinx(dw, depth)
    FF = dw + 10
    LUT = int(0.9687 * dw + 13.982)
  return {'BRAM': BRAM, 'DSP': DSP, 'FF': FF, 'LUT': LUT}

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
  parser.add_argument('-pr', '--hls-project', metavar='HLS_PROJECT', required=True, help='directory of HLS project')
  parser.add_argument('-p', '--platform', metavar='PLATFORM', required=True, help='hardware platform: intel/xilinx')

  args = parser.parse_args()

  if args.platform == 'intel':
    print('Intel platform not supported yet!')
  elif args.platform == 'xilinx':
    resource = xilinx_run(args.design_info_json, args.design_info_dat, args.hls_project)
    print("Est. FF: " + str(resource['FF']))
    print("Est. LUT: " + str(resource['LUT']))
    print("Est. BRAM: " + str(resource['BRAM']))
    print("Est. DSP: " + str(resource['DSP']))
