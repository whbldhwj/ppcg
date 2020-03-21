import sys
import argparse
import re

def print_module_def(f, arg_map, module_def, def_args, call_args_type):
  """Print out module definitions

  This function prints out the module definition with all arguments
  replaced by the calling arugments.

  Args:
    f: file handle
    arg_map: maps from module definition args to module call args
    module_def (list): stores the module definition texts
    def_args (list): stores the module definition arguments
    call_args_type (list): stores the type of each module call arg

  """
  # Extract module ids and fifos from def_args
  module_id_args = []
  fifo_args = []
  for i in range(len(def_args)):
    def_arg = def_args[i]
    arg_type = call_args_type[i]
    if arg_type == 'module id':
      module_id_args.append(def_arg)
    if arg_type == 'fifo':
      fifo_args.append(def_arg)

  for line in module_def:
    if line.find('__kernel') != -1:
      # This line is kernel argument.
      # All module id and fifo arguments are deleted
      m = re.search('(.+)\(', line)
      if m:
        prefix = m.group(1)
#        print(prefix)
      m = re.search('\((.+?)\)', line)
      if m:
        def_args = m.group(1)
      def_args = def_args.split(', ')
      new_def_args = []
      for i in range(len(def_args)):
        if call_args_type[i] != 'module id' and call_args_type[i] != 'fifo':
          new_def_args.append(def_args[i])
      f.write(prefix + '(')
      first = True
      for arg in new_def_args:
        if not first:
          f.write(', ')
        f.write(arg)
        first = False
      f.write(')\n')
    elif line.find('// module id') != -1:
      # This line is module id initialization
      # All module ids are replaced by call args
      for i in range(len(module_id_args)):
        def_arg = module_id_args[i]
        call_arg = arg_map[def_arg]
        line = line.replace(def_arg, call_arg)
      f.write(line)
    elif line.find('read_channel_intel') != -1 or line.find('write_channel_intel') != -1:
      # This line is fifo read/write
      # All fifo name is replaced by call args
      for i in range(len(fifo_args)):
        def_arg = fifo_args[i]
        call_arg = arg_map[def_arg]
        line = line.replace(def_arg, call_arg)
      f.write(line)
    else:
      f.write(line)

def generate_intel_kernel(kernel, headers, module_defs, module_calls, fifo_decls):
  with open(kernel, 'w') as f:
    # print out headers
    for header in headers:
      f.write(header + '\n')
    f.write('\n')

    # print out channels
    f.write('/* Channel Declaration */\n')
    for fifo_decl in fifo_decls:
      f.write(fifo_decl + '\n')
    f.write('/* Channel Declaration */\n\n')

    # print out module definitions
    for module_call in module_calls:
      f.write('/* Module Definition */\n')
      def_args = []
      call_args = []
      call_args_type = []
      arg_map = {}
      # Extract the module name
      line = module_call[0]
      m = re.search('(.+?)\(', line)
      if m:
        module_name = m.group(1)
      module_def = module_defs[module_name]
      # extract the arg list in module definition
      for line in module_def:
        if line.find('__kernel') != -1:
          m = re.search('\((.+?)\)', line)
          if m:
            def_args_old = m.group(1)
      def_args_old = def_args_old.split(', ')
      for arg in def_args_old:
        arg = arg.split()[-1]
        def_args.append(arg)

      # extract the arg list in module call
      for line in module_call:
        m = re.search('/\*(.+?)\*/', line)
        if m:
          arg_type = m.group(1).strip()
          call_args_type.append(arg_type)
          n = re.search('\*/ (.+)', line)
          if n:
            call_arg = n.group(1).strip(',')
            call_args.append(call_arg)

      # build a mapping between the def_arg to call_arg
      for i in range(len(def_args)):
        call_arg_type = call_args_type[i]
        if call_arg_type == 'module id' or call_arg_type == 'fifo':
          def_arg = def_args[i]
          call_arg = call_args[i]
          arg_map[def_arg] = call_arg

      # print out the module definition with call args plugged in
      print_module_def(f, arg_map, module_def, def_args, call_args_type)
      f.write('/* Module Definition */\n\n')

def insert_xlnx_pragmas(lines):
  """ Insert HLS pragmas for Xilinx program

  Replace the comments of "// hls_pipeline" and "// hls_unroll" with
  HLS pragmas
  For "// hls_pipeline", if the next codeline contains for loop, insert
  the "#pramga HLS PIPELINE II=1" below the for loop;
  otherwise, insert the pragma in-place.
  For "// hls_unroll", if the next codeline contains for loop, insert
  the "#pragma HLS UNROLL" below the for loop;
  otherwise, do not insert the pragma.

  Args:
    lines: contains the codelines of the program
  """

  code_len = len(lines)
  for pos in range(code_len):
    line = lines[pos]
    if line.find("// hls_pipeline") != -1:
      # check the next line
      next_line = lines[pos + 1]
      if next_line.find("for") != -1:
        indent = next_line.find("for")
        new_line = " " * indent + "#pragma HLS PIPELINE II=1\n"
        lines.insert(pos + 2, new_line)
        # delete the annotation
        del lines[pos]
      else:
        # insert the pragma in-place
        indent = line.find("//")
        new_line = " " * indent + "#pragma HLS PIPELINE II=1\n"
        del lines[pos]
        lines.insert(pos, new_line)
    elif line.find("// hls_unroll") != -1:
      # check the next line
      next_line = lines[pos + 1]
      if next_line.find("for") != -1:
        indent = next_line.find("for")
        new_line = " " * indent + "#pragma HLS UNROLL\n"
        lines.insert(pos + 2, new_line)
        # delete the annotation
        del lines[pos]


  return lines

def xilinx_run(kernel_call, kernel_def, kernel='kernel'):
  """ Generate kernel file for Xilinx platform

  We will copy the content of kernel definitions before the kernel calls.

  Args:
    kernel_call: file contains kernel calls
    kernel_def: file contains kernel definitions
    kernel: output kernel file

  """

  # Load kernel definition file
  lines = []
  with open(kernel_def, 'r') as f:
    lines = f.readlines()

  # Insert the HLS pragmas
  lines = insert_xlnx_pragmas(lines)

  kernel = str(kernel)
  kernel += '_xilinx.cpp'
  print("Please find the generated file: " + kernel)

  with open(kernel, 'w') as f:
    f.writelines(lines)
    with open(kernel_call, 'r') as f2:
      lines = f2.readlines()
      f.writelines(lines)

def intel_run(kernel_call, kernel_def, kernel='kernel'):
  """ Generate kernel file for Intel platform

  We will exrtract all teh fifo declarations and module calls.
  Then plut in the module definitions into each module call.

  Args:
    kernel_call: file contains kernel calls
    kernel_def: file contains kernel definitions
    kernel: output kernel file
  """

  # Load kernel call file
  module_calls = []
  fifo_decls = []
  with open(kernel_call, 'r') as f:
    add = False
    while True:
      line = f.readline()
      if not line:
        break
      # Extract the fifo declaration and add to the list
      if add:
        line = line.strip()
        fifo_decls.append(line)
      if line.find('/* FIFO Declaration */') != -1:
        if add:
          fifo_decls.pop(len(fifo_decls) - 1)
        add = not add

  with open(kernel_call, 'r') as f:
    add = False
    module_call = []
    while True:
      line = f.readline()
      if not line:
        break
      # Extract the module call and add to the list
      if add:
        line = line.strip()
        module_call.append(line)
      if line.find('/* Module Call */') != -1:
        if add:
          module_call.pop(len(module_call) - 1)
          module_calls.append(module_call.copy())
          module_call.clear()
        add = not add

  module_defs = {}
  headers = []
  with open(kernel_def, 'r') as f:
    while True:
      line = f.readline()
      if not line:
        break
      if line.find('#include') != -1:
        line = line.strip()
        headers.append(line)

  with open(kernel_def, 'r') as f:
    add = False
    module_def = []
    while True:
      line = f.readline()
      if not line:
        break
      # Extract the module definition and add to the dict
      if add:
        module_def.append(line)
        # Extract the module name
        if (line.find('__kernel')) != -1:
          m = re.search('void (.+?)\(', line)
          if m:
            module_name = m.group(1)
      if line.find('/* Module Definition */') != -1:
        if add:
          module_def.pop(len(module_def) - 1)
          module_defs[module_name] = module_def.copy()
          module_def.clear()
        add = not add

  # compose the kernel file
  kernel = str(kernel)
  kernel += '_intel.c'
  generate_intel_kernel(kernel, headers, module_defs, module_calls, fifo_decls)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='==== PolySA CodeGen ====')
  parser.add_argument('-c', '--kernel-call', metavar='KERNEL_CALL', required=True, help='kernel function call')
  parser.add_argument('-d', '--kernel-def', metavar='KERNEL_DEF', required=True, help='kernel function definition')
  parser.add_argument('-p', '--platform', metavar='PLATFORM', required=True, help='hardware platform: intel/xilinx')
  parser.add_argument('-k', '--kernel', metavar='KERNEL', required=False, default='kernel', help='output kernel file')

  args = parser.parse_args()

  if args.platform == 'intel':
    intel_run(args.kernel_call, args.kernel_def, args.kernel)
  elif args.platform == 'xilinx':
    xilinx_run(args.kernel_call, args.kernel_def, args.kernel)

