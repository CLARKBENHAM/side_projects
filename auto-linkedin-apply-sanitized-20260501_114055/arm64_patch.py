"""Patch for Apple Silicon (ARM64) Macs where objc_msgSend variants are missing."""
import ctypes
import platform

if platform.machine() == "arm64":
    _orig_getattr = ctypes.CDLL.__getattr__

    def _patched_getattr(self, name):
        try:
            return _orig_getattr(self, name)
        except AttributeError:
            if "objc_msgSend" in name:
                return _orig_getattr(self, "objc_msgSend")
            raise

    ctypes.CDLL.__getattr__ = _patched_getattr  # type: ignore[assignment]
