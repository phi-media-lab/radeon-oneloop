# CPU-edge runtime

The edge process handles camera and joint acquisition, sequence IDs, command
timeouts, joint/action limits, watchdog, emergency stop, and robot I/O. It must
not execute policy inference on a second GPU or NPU in the formal profile.
