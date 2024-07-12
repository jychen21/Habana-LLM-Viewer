# Todo

'''
Example Roofline Calculation
Assume:

Total Data Moved: \text{size} \times \text{num_procs} \times (\text{num_procs} - 1)
Elapsed Time: Measured from the script
Bandwidth (BW): Given or measured in GB/s
Using these values, you can plot the Roofline model by comparing the measured performance against the theoretical maximum bandwidth and latency.

For a comprehensive Roofline analysis, including computation and communication, you might need to combine the results from computational kernels and communication patterns, plotting them together to identify bottlenecks.
'''


def allgather(config, transfer_size):
    pass


def allreduce(config, transfer_size):
    pass
