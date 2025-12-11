# benign_script.py - example benign behavior
import socket, time, os
open('/tmp/bfzzd_test.txt','w').write('hello')
s = socket.socket()
try:
    s.bind(('127.0.0.1',0))
    s.close()
except Exception:
    pass
time.sleep(0.2)
