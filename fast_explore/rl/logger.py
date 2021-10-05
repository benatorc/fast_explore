import json
import time
import numpy as np

from pathlib import Path
from tensorboardX import SummaryWriter

class Logger:
  def __init__(self, logdir, step, logname='metrics.jsonl'):
    if not isinstance(logdir, Path):
      logdir = Path(logdir)
    self._logdir = logdir
    self._logname = logname
    self._writer = SummaryWriter(str(logdir))
    self._last_step = None
    self._last_time = None
    self._scalars = {}
    self._strings = {}
    self._images = {}
    self._videos = {}
    self._histograms = {}
    self.step = step

    self._first_time = time.time()

  def scalar(self, name, value):
    self._scalars[name] = float(value)
  
  # log generated molecules
  def string(self, name, string):
    self._strings[name] = string

  def image(self, name, value):
    self._images[name] = np.array(value)

  def video(self, name, value):
    self._videos[name] = np.array(value)
    
  def histogram(self, name, value):
    self._histograms[name] = np.array(value)

  def write(self, fps=False):
    scalars = list(self._scalars.items())
    if fps:
      scalars.append(('fps', self._compute_fps(self.step)))
    print(f'[{self.step}]', ' / '.join(f'{k} {v:.1f}' for k, v in scalars))
    metrics_filename = (self._logdir / self._logname)
    with metrics_filename.open('a') as f:
      f.write(json.dumps({'step': self.step, ** dict(scalars + list(self._strings.items()))}) + '\n')
    for name, value in scalars:
      self._writer.add_scalar('scalars/' + name, value, self.step)
    for name, value in self._images.items():
      self._writer.add_image(name, value, self.step, dataformats='HWC')
    for name, value in self._videos.items():
      self._writer.add_video(name, value, self.step)
    for name, value in self._histograms.items():
      self._writer.add_histogram(name, value, self.step)
    self._writer.flush()
    self._scalars = {}
    self._images = {}
    self._videos = {}
    self._histograms = {}

  def get_total_time(self):
      # Returns the total amount of time spent in minutes
      return (time.time() - self._first_time) / (60)

  def _compute_fps(self, step):
    if self._last_step is None:
      self._last_time = time.time()
      self._last_step = step
      return 0
    steps = step - self._last_step
    duration = time.time() - self._last_time
    self._last_time += duration
    self._last_step = step
    return steps / duration