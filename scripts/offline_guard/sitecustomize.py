"""Defense against accidental Python network calls in the offline check suite.

CI additionally uses Docker --network none, which also covers subprocesses that
replace PYTHONPATH and non-Python tools. This Python guard is not a sandbox.
"""
import os
import socket
import sys


def deny_network(event, arguments):
    if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}:
        raise OSError("Network access is disabled during RSI offline checks")
    if event == "socket.connect" and arguments[0].family in {socket.AF_INET, socket.AF_INET6}:
        raise OSError("Network access is disabled during RSI offline checks")


if os.environ.get("RSI_OFFLINE_CHECK") == "1":
    sys.addaudithook(deny_network)
