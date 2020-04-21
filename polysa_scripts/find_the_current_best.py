from optimizer import add_cycle_dse_info
import argparse

def find_the_best(n_job):
  design_infos = []
  if n_job == 1:
    prj_dir = 'polysa.tmp/optimizer/search'
    add_cycle_dse_info(prj_dir, design_infos)
  else:
    prj_dir = 'polysa.tmp/optimizer/search'
    for job in range(n_job):
      job_prj_dir = prj_dir + '/job'
      job_prj_dir += str(job)
      add_cycle_dse_info(job_prj_dir, design_infos)

  design_infos.sort(key=lambda x:x['latency'], reverse=False)
  print('best latency: ', design_infos[0]['latency'])
  print('best resource: ', design_infos[0]['resource'])
  print('dir: ', design_infos[0]['dir'])

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='== Find the Best Design ==')
  parser.add_argument('-j', '--job', required=True, help='number of parallel jobs')

  args = parser.parse_args()
  find_the_best(int(args.job))
