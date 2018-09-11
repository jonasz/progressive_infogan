import tempfile, os, subprocess, shutil, sys, getopt, random
import time
import tensorflow as tf

LATEST_EVENTS = 100

def syscmd(cmd):
  res = os.system(cmd)
  assert res == 0, 'CMD: %s, RES: %d' % (cmd, res)

def parse_lines(content):
  lines = content.split('\n')
  lines = map(lambda x: x.strip(), lines)
  lines = filter(None, lines)
  return lines

def promptuser(lines):
	"""Display <lines> to user in their favourite editor.
	Then return the lines he entered ommiting empty lines and
	ones beginning with a #."""

	tf = tempfile.NamedTemporaryFile(delete=False)
	for line in lines:
		tf.write(line+'\n')
	tf.close()
	editor = os.getenv('EDITOR')
	if editor==None: 
		editor = 'vi'
	subp = subprocess.Popen([editor, tf.name])
	subp.wait()
	f = open(tf.name)
	res = f.readlines()
	f.close()
	os.remove(tf.name)
	res= map( lambda x: x.strip(), res) #strip
	res= filter(None, res) #remove empty lines
	res= filter( lambda x: x[0]!='#', res) #remove comment lines
	return res

def get_job_ids():
  #  return ['job_41fef1f_20180708_193249', 'job_093e6d6_20180708_184417']
  lines = subprocess.check_output('gcloud ml-engine jobs list'.split(' '))
  lines = parse_lines(lines)

  interesting_jobs = []
  for i, raw_line in enumerate(lines):
    line = raw_line.split(' ')
    line = map(lambda x: x.strip(), line)
    line = filter(None, line)
    job_id, status, timestamp = line
    if job_id == 'JOB_ID': continue
    #  print job_id, status, timestamp
    if status in ('RUNNING', 'SUCCEEDED') or i < 5:
      interesting_jobs.append('%s  # %s' % (job_id, status))
  res = promptuser(interesting_jobs[:8])
  res = map(lambda x: x.split(' ')[0], res)
  return res

def download_last_events(base_dir, job_id):
  target_dir = os.path.join(base_dir, job_id)
  tf.gfile.MakeDirs(target_dir)
  os.chdir(target_dir)

  # ./event*
  events = parse_lines(subprocess.check_output(
      ('gsutil ls gs://seer-of-visions-ml2/%s/event*' % job_id).split(' ')))
  events = events[-LATEST_EVENTS:]
  syscmd('gsutil -m cp %s .' % ' '.join(events))

  # ./eval/event*
  tf.gfile.MakeDirs('./eval')
  os.chdir('./eval')
  events = parse_lines(subprocess.check_output(
      ('gsutil ls gs://seer-of-visions-ml2/%s/eval/event*' % job_id).split(' ')))
  events = events[-LATEST_EVENTS:]
  syscmd('gsutil -m cp %s .' % ' '.join(events))


def main():
  job_ids = get_job_ids()
  base_dir = '/tmp/tensorboard_gcloud_%s' % int(time.time())
  tf.gfile.MakeDirs(base_dir)
  time.sleep(1.)
  #  base_dir = '/tmp/tensorboard_gcloud_1531128020'
  os.chdir(base_dir)
  for job_id in job_ids:
    print 'Downloading %s' % job_id
    try:
      download_last_events(base_dir, job_id)
    except Exception, e:
      print 'FAIL'
      print e
  os.chdir(base_dir)

  os.execlp('tensorboard', 'tensorboard', '--port', '5084', '--logdir', '.')


if __name__ == '__main__':
  main()
