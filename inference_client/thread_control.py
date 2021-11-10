# 全局控制active
import ctypes
import inspect


def _init():
    global active
    active = {}


def get_device_status(device):
    global active
    if 'device' + device in active.keys():
        return active['device' + device]
    else:
        return False


def set_device_status(device, status):
    global active
    active['device' + str(device)] = status


def _async_raise(tid, exctype):
    """Raises an exception in the threads with id tid"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    print('res:', res)
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


_init()
