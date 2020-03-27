import logging
import datetime
import sys
import os

def loggersetup(log_dir, stdout_level = logging.DEBUG, file_level = logging.INFO):
  # Define default logfile format.
  file_name_format = '{year:04d}{month:02d}{day:02d}-'\
      '{hour:02d}{minute:02d}{second:02d}.log'

  # Define the default logging message formats.
  file_msg_format = '%(asctime)s %(levelname)-8s: %(message)s'
  console_msg_format = '%(levelname)s: %(message)s'

  # Define the log rotation criteria.
  max_bytes=1024**2
  backup_count=100

  # Validate the given directory.
  log_dir = os.path.normpath(log_dir)

  # Create a folder for the logfiles.
  if not os.path.exists(log_dir):
      os.makedirs(log_dir)

  # Define default logfile format.
  file_name_format = '{year:04d}{month:02d}{day:02d}.log'

  # Construct the name of the logfile.
  t = datetime.datetime.now()
  file_name = file_name_format.format(year=t.year, month=t.month, day=t.day,
      hour=t.hour, minute=t.minute, second=t.second)
  file_name = os.path.join(log_dir, file_name)

  # Change root logger level from WARNING (default) to NOTSET in order for all messages to be delegated.
  logging.getLogger().setLevel(logging.NOTSET)
  logging.getLogger().handlers = [] # remove old handlers

  # Add stdout handler, with level INFO
  console = logging.StreamHandler(sys.stdout)
  console.setLevel(logging.DEBUG)
  formater = logging.Formatter('[%(levelname)s] %(message)s')
  console.setFormatter(formater)
  logging.getLogger().addHandler(console)

  # Add file rotating handler, with level DEBUG
  fileHandler = logging.FileHandler(file_name)
  fileHandler.setLevel(logging.DEBUG)
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
  fileHandler.setFormatter(formatter)
  logging.getLogger().addHandler(fileHandler)

  log = logging.getLogger(__name__)
  return log