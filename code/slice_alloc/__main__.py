"""Single entry point for the complete low-altitude slice study."""
import argparse
import shutil
import subprocess
import sys

def main():
 parser=argparse.ArgumentParser(description='低空多模态网络切片：完整验证管线')
 parser.add_argument('command',nargs='?',default='verify',choices=['build','prepare','train','evaluate','all','verify','report','finalize'])
 parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
 from slice_alloc import study as s
 if args.command=='build':
  for source,target in [('allocation_final_scenario.cc','allocation-final-scenario.cc'),('slice_queue.hpp','slice_queue.hpp')]:shutil.copy2(s.PACKAGE/'execution'/source,s.b.p.NS3/'scratch'/target)
  subprocess.run([sys.executable,'waf','build','--targets','allocation-final-scenario','-j','4'],cwd=s.b.p.NS3,check=True);return
 if args.command in ('verify','report'):
  from slice_alloc.tools.study_report import report
  report(args.workers)
 elif args.command=='finalize':
  from slice_alloc.tools.study_finalize import finalize
  finalize()
 else:
  from slice_alloc.tools.study_run import main as run
  sys.argv=[sys.argv[0],args.command,'--workers',str(args.workers)];run()

if __name__=='__main__':main()
