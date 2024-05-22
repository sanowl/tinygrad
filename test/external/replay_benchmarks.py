import subprocess
import os

TRACE_DIR = os.path.join(os.environ["HOME"], "traces")
ret = subprocess.run(['ls', '-l', TRACE_DIR], capture_output=True, text=True)
print(len(ret.stdout))
