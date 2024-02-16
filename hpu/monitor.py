import pyhlml
import time
import pdb
from ctypes import *
import ctypes

from prometheus_client import start_http_server
from prometheus_client import Gauge



# This function assumes that the string buffer is a contiguous array of bytes and the number of strings is known.
def extract_strings(str_buf, noc):
    extracted_strings = []
    offset = 0

    for _ in range(noc):
        char_list = []
        while (ord(str_buf[offset])) == 0:  # Null byte check
            # Ensure we're working with an integer representation of the byte
            char = str_buf[offset]
            if isinstance(char, bytes):
                char = ord(char)
            offset += 1
        while True:
            char = str_buf[offset]
            # Ensure we're working with an integer representation of the byte
            if isinstance(char, bytes):  # This check might need adjusting
                char = ord(char)  # Convert bytes to integer if necessary

            if char == 0:  # Null byte check
                break

            char_list.append(char)
            offset += 1

        # Directly convert the list of integers to bytes and then decode
        extracted_string = bytes(char_list).decode('ascii')
        extracted_strings.append(extracted_string)

        offset += 1  # Skip over the null terminator

    return extracted_strings

# Initialize the library
pyhlml.hlmlInit()

# Get the number of devices
nb_devices = pyhlml.hlmlDeviceGetCount()
print(f'nb_devices : {nb_devices}')

g1 = Gauge('gpu_utilization', 'GPU Utilization', ['device'])
g2 = Gauge('memory_utilization', 'Memory Utilization', ['device'])
g3 = Gauge('pcie_throughput', 'PCIE Throughput', ['device','direction'])

# Get the handle of each device
devices = []
for i in range(nb_devices):
    d = pyhlml.hlmlDeviceGetHandleByIndex(i)
    if d:
        devices.append(d)
        print(f'device ({i}) type {pyhlml.hlmlDeviceGetName(d)} : {pyhlml.hlmlDeviceGetPCIInfo(d)}')
        g1.labels(device=f'gpu{i}')
        g2.labels(device=f'gpu{i}')
        g3.labels(device=f'gpu{i}',direction='tx')
        g3.labels(device=f'gpu{i}',direction='rx')

# Start the Prometheus server
start_http_server(9100)

extracted_strings = []
first_loop = True
while True:
    i = 0
    start_time = time.time()
    # Get the memory info and utilization rates of each device
    for d in devices:
        try:
            _m = pyhlml.hlmlDeviceGetMemoryInfo(d)
            _u = pyhlml.hlmlDeviceGetUtilizationRates(d)
            _ptx = pyhlml.hlmlDeviceGetPCIEThroughput(d,0)
            _prx = pyhlml.hlmlDeviceGetPCIEThroughput(d,1)
            g1.labels(device=f'gpu{i}').set(_u)
            g2.labels(device=f'gpu{i}').set(_m.used/_m.total)
            g3.labels(device=f'gpu{i}', direction='tx').set(_ptx)
            g3.labels(device=f'gpu{i}', direction='rx').set(_prx)
            print(f'({i}) usage : {_u}, memory_info : {_m.total} {_m.used} , pcie_throughput : {_ptx} {_prx}')
        except:
            print(f'({d} failed to get usage')
        # _mac = pyhlml.hlmlDeviceGetMacAddrInfo(d)
        # _n = _mac[0][0]
        if first_loop:
            nic_gauge = Gauge('nic_stats', 'NIC Stats', ['device','port','stat'])
            for _i in range(24):
                try:
                    _link = pyhlml.hlmlDeviceNicGetLink(d,_i)
                    if _link == False:
                        print(f'port {_i} is not connected')
                        continue
                    _stats = pyhlml.hlmlDeviceNicGetStatistics(d,_i)
                    _noc = _stats.num_of_counters_out[0]
                    extracted_strings = extract_strings(_stats.str_buf, _noc)
                    for _n in range(_noc):
                        nic_gauge.labels(device=f'gpu{i}',port=f'port{_i}',stat=extracted_strings[_n]).set(_stats.val_buf[_n])
                except:
                    continue
            first_loop = False
        # Get stats for each port of the device
        for _i in range(24):
            try:
                _link = pyhlml.hlmlDeviceNicGetLink(d,_i)
                if _link == False:
                    continue
                _stats = pyhlml.hlmlDeviceNicGetStatistics(d,_i)
                _noc = _stats.num_of_counters_out[0]
                # extracted_strings = extract_strings(_stats.str_buf, _noc)
                for _n in range(_noc):
                    nic_gauge.labels(device=f'gpu{i}',port=f'port{_i}',stat=extracted_strings[_n]).set(_stats.val_buf[_n])
            except:
                continue
        i += 1
    elapsed_time = time.time() - start_time
    print(f'elapsed_time : {elapsed_time}')
    adjust_time = 20-elapsed_time
    if adjust_time > 0:
        time.sleep(20-elapsed_time)
    # print('')