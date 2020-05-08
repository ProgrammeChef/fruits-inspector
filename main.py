#!/usr/bin/env python
"""
Main execution file.
"""

import first_task
import second_task
import final_challenge

__author__ = "Marco Rossini"
__copyright__ = "Copyright 2020, Marco Rossini"
__date__ = "2020/05"
__license__ = "MIT"
__version__ = "1.0"

# ----------------------------------------------------------------------------------------------------------------------

# Run first task
print("----- Running first task -----")
first_task.run()

# Run second task
print("\n----- Running second task -----")
second_task.run_clustering()
second_task.run_samples()

# Run final challenge
print("\n----- Running final challenge -----")
final_challenge.run()
