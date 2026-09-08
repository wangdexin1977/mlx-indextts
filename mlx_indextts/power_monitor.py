"""Native macOS sleep/wake notifications without third-party dependencies."""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable


_IOKIT_MESSAGE_BASE = 0xE0000000
K_IO_MESSAGE_CAN_SYSTEM_SLEEP = _IOKIT_MESSAGE_BASE | 0x270
K_IO_MESSAGE_SYSTEM_WILL_SLEEP = _IOKIT_MESSAGE_BASE | 0x280
K_IO_MESSAGE_SYSTEM_HAS_POWERED_ON = _IOKIT_MESSAGE_BASE | 0x300


class MacOSPowerMonitor:
    """Deliver system sleep/wake events from IOKit on a dispatch queue."""

    def __init__(
        self,
        on_sleep: Callable[[], None],
        on_wake: Callable[[], None],
    ) -> None:
        self._on_sleep = on_sleep
        self._on_wake = on_wake
        self._iokit: ctypes.CDLL | None = None
        self._callback = None
        self._notification_port = ctypes.c_void_p()
        self._notifier = ctypes.c_uint(0)
        self._root_port = 0

    def start(self) -> bool:
        """Register the monitor. Return False when unavailable or registration fails."""
        if sys.platform != "darwin":
            return False

        try:
            iokit = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
            dispatch = ctypes.CDLL("/usr/lib/system/libdispatch.dylib")

            callback_type = ctypes.CFUNCTYPE(
                None,
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_uint32,
                ctypes.c_void_p,
            )

            def power_callback(
                _refcon: int,
                _service: int,
                message_type: int,
                message_argument: int,
            ) -> None:
                try:
                    if message_type == K_IO_MESSAGE_SYSTEM_WILL_SLEEP:
                        self._on_sleep()
                    elif message_type == K_IO_MESSAGE_SYSTEM_HAS_POWERED_ON:
                        self._on_wake()
                except Exception as exc:
                    print(f"Warning: macOS power event callback failed: {exc}")
                finally:
                    if message_type in {
                        K_IO_MESSAGE_CAN_SYSTEM_SLEEP,
                        K_IO_MESSAGE_SYSTEM_WILL_SLEEP,
                    }:
                        notification_id = int(message_argument or 0)
                        iokit.IOAllowPowerChange(self._root_port, notification_id)

            self._callback = callback_type(power_callback)
            iokit.IORegisterForSystemPower.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                callback_type,
                ctypes.POINTER(ctypes.c_uint),
            ]
            iokit.IORegisterForSystemPower.restype = ctypes.c_uint
            iokit.IOAllowPowerChange.argtypes = [ctypes.c_uint, ctypes.c_ssize_t]
            iokit.IOAllowPowerChange.restype = ctypes.c_int
            iokit.IONotificationPortSetDispatchQueue.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            iokit.IONotificationPortSetDispatchQueue.restype = None
            dispatch.dispatch_get_global_queue.argtypes = [ctypes.c_long, ctypes.c_ulong]
            dispatch.dispatch_get_global_queue.restype = ctypes.c_void_p

            self._root_port = int(
                iokit.IORegisterForSystemPower(
                    None,
                    ctypes.byref(self._notification_port),
                    self._callback,
                    ctypes.byref(self._notifier),
                )
            )
            if not self._root_port or not self._notification_port.value:
                return False

            queue = dispatch.dispatch_get_global_queue(0, 0)
            if not queue:
                return False
            iokit.IONotificationPortSetDispatchQueue(self._notification_port, queue)
            self._iokit = iokit
            return True
        except (AttributeError, OSError, TypeError, ValueError):
            return False


def start_macos_power_monitor(
    on_sleep: Callable[[], None],
    on_wake: Callable[[], None],
) -> MacOSPowerMonitor | None:
    """Start a retained monitor, returning None if native monitoring is unavailable."""
    monitor = MacOSPowerMonitor(on_sleep, on_wake)
    return monitor if monitor.start() else None
