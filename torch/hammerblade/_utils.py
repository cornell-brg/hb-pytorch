import torch
import torch._six


def _get_device_index(device, optional=False):
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a HammerBlade device. Note that for a HammerBlade device without a specified index,
    i.e., ``torch.device('cuda')``, this will return the current default HammerBlade
    device if :attr:`optional` is ``True``.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default HammerBlade
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, torch._six.string_classes):
        device = torch.device(device)
    if isinstance(device, torch.device):
        dev_type = device.type
        if device.type != 'hammerblade':
            raise ValueError('Expected a hammerblade device, but got: {}'.format(device))
        device_idx = device.index
    else:
        device_idx = device
    if device_idx is None:
        if optional:
            # default hammerblade device index
            return torch.hammerblade.current_device()
        else:
            raise ValueError('Expected a hammerblade device with a specified index '
                             'or an integer, but got: '.format(device))
    return device_idx
