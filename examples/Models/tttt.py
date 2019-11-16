from os import getpid, kill
from time import sleep
import re
import signal

from notebook.notebookapp import list_running_servers
from requests import get
from requests.compat import urljoin
import ipykernel
import json
import psutil


def get_active_kernels(cpu_threshold):
    """Get a list of active jupyter kernels."""
    active_kernels = []
    pids = psutil.pids()
    my_pid = getpid()

    for pid in pids:
        if pid == my_pid:
            continue
        try:
            p = psutil.Process(pid)
            cmd = p.cmdline()
            for arg in cmd:
                if arg.count('ipykernel'):
                    cpu = p.cpu_percent(interval=0.1)
                    if cpu > cpu_threshold:
                        active_kernels.append((cpu, pid, cmd))
        except psutil.AccessDenied:
            continue
    return active_kernels


def interrupt_bad_notebooks(cpu_threshold=0.2):
    """Interrupt active jupyter kernels. Prompts the user for each kernel."""

    active_kernels = sorted(get_active_kernels(cpu_threshold), reverse=True)

    servers = list_running_servers()
    for ss in servers:
        response = get(urljoin(ss['url'].replace('localhost', '127.0.0.1'), 'api/sessions'),
                       params={'token': ss.get('token', '')})
        for nn in json.loads(response.text):
            for kernel in active_kernels:
                for arg in kernel[-1]:
                    if arg.count(nn['kernel']['id']):
                        pid = kernel[1]
                        cpu = kernel[0]
                        interrupt = input(
                            'Interrupt kernel {}; PID: {}; CPU: {}%? (y/n) '.format(nn['notebook']['path'], pid, cpu))
                        if interrupt.lower() == 'y':
                            p = psutil.Process(pid)
                            while p.cpu_percent(interval=0.1) > cpu_threshold:
                                kill(pid, signal.SIGINT)
                                sleep(0.5)

if __name__ == '__main__':
    interrupt_bad_notebooks()