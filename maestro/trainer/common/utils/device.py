import torch


def parse_device_spec(device_spec: str | torch.device) -> torch.device:
    """
    Convert a string or torch.device into a valid torch.device. Optimized by precompiling
    regex and efficient device querying.

    Args:
        device_spec (str | torch.device): A specification for the device.

    Returns:
        torch.device: The corresponding torch.device object.

    Raises:
        ValueError: If the device specification is unrecognized or the GPU index is out of range.
    """
    if isinstance(device_spec, torch.device):
        return device_spec

    device_str = device_spec.lower()
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    match = _cuda_device_regex.match(device_str)
    if match:
        index = int(match.group(1))
        device_count = torch.cuda.device_count()
        if index < 0 or index >= device_count:
            raise ValueError(
                f"Requested cuda:{index} but only {device_count} GPU(s) are available."
            )
        return torch.device(f"cuda:{index}")

    if device_str in {"cpu", "cuda", "mps"}:
        return torch.device(device_str)

    raise ValueError(f"Unrecognized device spec: {device_spec}")


def device_is_available(device: torch.device) -> bool:
    """
    Check whether a given torch.device is available on the current system.

    Args:
        device (torch.device): The device to verify.

    Returns:
        bool: True if the device is available, False otherwise.
    """
    if device.type == "cuda":
        return torch.cuda.is_available()
    elif device.type == "mps":
        return torch.backends.mps.is_available()
    elif device.type == "cpu":
        return True
    return False
