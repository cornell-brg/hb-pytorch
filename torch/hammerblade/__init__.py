import contextlib
import platform
import ctypes
import os
import sys
import torch
import traceback
import warnings
import threading
from torch._six import raise_from
from subprocess import Popen, PIPE
from multiprocessing.util import register_after_fork as _register_after_fork
from ._utils import _get_device_index

_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls = []  # don't invoke these until initialization occurs
_in_bad_fork = False  # this global is also used in torch.manual_seed
_original_pid = False

class DeferredHammerBladeCallError(Exception):
    pass


def init():
    r"""Initialize PyTorch's HammerBlade state.  You may need to call
    this explicitly if you are interacting with PyTorch via
    its C API, as Python bindings for HB functionality will not
    be until this initialization takes place.  Ordinary users
    should not need this, as all of PyTorch's HB methods
    automatically initialize HB state on-demand.

    Does nothing if the HB state is already initialized.
    """
    _lazy_init()


def _lazy_init():
    global _initialized, _original_pid, _queued_calls
    if _initialized or hasattr(_tls, 'is_initializing'):
        return
    with _initialization_lock:
        # We be double-checked locking, boys!  This is OK because
        # the above test was GIL protected anyway.  The inner test
        # is for when a thread blocked on some other thread which was
        # doing the initialization; when they get the lock, they will
        # find there is nothing left to do.
        if _initialized:
            return
        # It is important to prevent other threads from entering _lazy_init
        # immediately, while we are still guaranteed to have the GIL, because some
        # of the C calls we make below will release the GIL
        if _in_bad_fork:
            raise RuntimeError(
                "Cannot re-initialize HammerBlade in forked subprocess.")
        # TODO: enable this
        # _check_driver()
        torch._C._hammerblade_init()
        _original_pid = os.getpid()
        # Some of the queued calls may reentrantly call _lazy_init();
        # we need to just return without initializing in that case.
        # However, we must not let any *other* threads in!
        _tls.is_initializing = True
        try:
            for queued_call, orig_traceback in _queued_calls:
                try:
                    queued_call()
                except Exception as e:
                    msg = ("HammerBlade call failed lazily at initialization with error: {}\n\n"
                           "HammerBlade call was originally invoked at:\n\n{}").format(str(e), orig_traceback)
                    raise_from(DeferredHammerBladeCallError(msg), e)
        finally:
            delattr(_tls, 'is_initializing')
        _initialized = True


def _after_fork(arg):
    global _initialized, _in_bad_fork
    if _initialized and _original_pid != os.getpid():
        _initialized = False
        _in_bad_fork = True
        _HammerBladeBase.__new__ = _lazy_new
        torch._C._hammerblade_set_run_yet_variable_to_false()

_register_after_fork(_after_fork, _after_fork)


class hammerbladeStatus(object):
    SUCCESS = 0
    ERROR_NOT_READY = 34


class HammerBladeError(RuntimeError):
    def __init__(self, code):
        super(HammerBladeError, self).__init__('({1})'.format(code))


def check_error(res):
    if res != hammerbladeStatus.SUCCESS:
        raise HammerBladeError(res)


class device(object):
    r"""Context-manager that changes the selected device.

    Arguments:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        assert self.idx == self.prev_idx
        # if self.idx == -1:
        #     return
        # self.prev_idx = torch._C._hammerblade_getDevice()
        # if self.prev_idx != self.idx:
        #     torch._C._hammerblade_setDevice(self.idx)
        _lazy_init()

    def __exit__(self, *args):
        assert self.idx == self.prev_idx
        #if self.prev_idx != self.idx:
        #    torch._C._hammerblade_setDevice(self.prev_idx)
        #return False


def current_device():
    r"""Returns the index of a currently selected device."""
    _lazy_init()
    return torch._C._hammerblade_getDevice()

################################################################################
# Define Storage and Tensor classes
################################################################################


from ..storage import _StorageBase


def _dummy_type(name):
    def init_err(self):
        class_name = self.__class__.__name__
        raise RuntimeError(
            "Tried to instantiate dummy base class {}".format(class_name))
    return type(storage_name, (object,), {"__init__": init_err})


if not hasattr(torch._C, 'HammerBladeDoubleStorageBase'):
    # Define dummy base classes
    for t in ['Double', 'Float', 'Long', 'Int', 'Short', 'Char', 'Byte', 'Half', 'Bool', 'BFloat16']:
        storage_name = 'HammerBlade{0}StorageBase'.format(t)
        tensor_name = 'HammerBlade{0}TensorBase'.format(t)

        torch._C.__dict__[storage_name] = _dummy_type(storage_name)
        torch._C.__dict__[tensor_name] = _dummy_type(tensor_name)


@staticmethod
def _lazy_new(cls, *args, **kwargs):
    _lazy_init()
    # We need this method only for lazy init, so we can remove it
    del _HammerBladeBase.__new__
    return super(_HammerBladeBase, cls).__new__(cls, *args, **kwargs)


class _HammerBladeBase(object):
    is_cuda = False
    is_hammerblade = True
    is_sparse = False

    def type(self, *args, **kwargs):
        with device(self.get_device()):
            return super(_HammerBladeBase, self).type(*args, **kwargs)

    __new__ = _lazy_new


class DoubleStorage(_HammerBladeBase, torch._C.HammerBladeDoubleStorageBase, _StorageBase):
    pass


class FloatStorage(_HammerBladeBase, torch._C.HammerBladeFloatStorageBase, _StorageBase):
    pass


class LongStorage(_HammerBladeBase, torch._C.HammerBladeLongStorageBase, _StorageBase):
    pass


class IntStorage(_HammerBladeBase, torch._C.HammerBladeIntStorageBase, _StorageBase):
    pass


class ShortStorage(_HammerBladeBase, torch._C.HammerBladeShortStorageBase, _StorageBase):
    pass


class CharStorage(_HammerBladeBase, torch._C.HammerBladeCharStorageBase, _StorageBase):
    pass


class ByteStorage(_HammerBladeBase, torch._C.HammerBladeByteStorageBase, _StorageBase):
    pass


class HalfStorage(_HammerBladeBase, torch._C.HammerBladeHalfStorageBase, _StorageBase):
    pass


class BoolStorage(_HammerBladeBase, torch._C.HammerBladeBoolStorageBase, _StorageBase):
    pass


class BFloat16Storage(_HammerBladeBase, torch._C.HammerBladeBFloat16StorageBase, _StorageBase):
    pass

torch._storage_classes.add(DoubleStorage)
torch._storage_classes.add(FloatStorage)
torch._storage_classes.add(LongStorage)
torch._storage_classes.add(IntStorage)
torch._storage_classes.add(ShortStorage)
torch._storage_classes.add(CharStorage)
torch._storage_classes.add(ByteStorage)
torch._storage_classes.add(HalfStorage)
torch._storage_classes.add(BoolStorage)
torch._storage_classes.add(BFloat16Storage)
