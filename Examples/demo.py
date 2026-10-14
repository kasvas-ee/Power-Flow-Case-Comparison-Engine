# ##########################################################################################
# A collection of standalone examples to demonstrate the features of the engine using various scenarios
# ##########################################################################################

import Examples.examples as examples

# -----------------------------------------------------
# 1. Demonstrate Engine Basics
# -----------------------------------------------------

# # Understand text report: a Matching case on a small system (IEEE 13 bus system)
# examples.Bus14Example.as_is()

# # Understand severity: a Mis-matching case on a small system (IEEE 13 bus system)
# examples.Bus14Example.output_mismatch()

# -----------------------------------------------------
# 2. Demonstrate Pre-checks
# -----------------------------------------------------

# # Compare two powerflow files generated from 2 completeld different models.
# examples.SillyExample.compare_diff_models()

# # Topology mismatch i.e. the bus count matches but connectivity between buses does not match
# examples.Bus14Example.topology_mismatch()

# non-bus asset count mismatch
# examples.Bus14Example.count_mismatch()

# # What if a metric is missing
# examples.Bus14Example.missing_metric()

# # Ybus has identical pattern but values have a mismatch (i.e. model comparable but not identical)
# # Discuss scalability of TODO assets (here it's sgen)
# examples.Bus300Example.as_is()

# -----------------------------------------------------
# 3. Demonstrate Insights
# -----------------------------------------------------

# # Ideal condition
# examples.Bus39Example.as_is()

# # Relative pattern: All loads scaled by 25%
# examples.Bus39Example.loads_doubled()

# # Very-few very-large Mismatch: 5% of loads scaled by 20%, rest by 1%
# examples.Bus118Example.extreme_outliner()

# # Just a few mismatches: 5% of loads scaled by 5%
# examples.Bus118Example.concentrated_mismatch()

# Some loads scaled down, others scaled up: 50% loads scaled up by 10%, remaining scaled down by 10%.
examples.Bus118Example.directional_pattern()

# -----------------------------------------------------
# 4. Demonstrate Parallel processing
# -----------------------------------------------------
# NOT IMPLEMENTED!!
# examples.case2869pegase.as_is()
# Parallel process time diff: Check on a 2869 bus system


print("That's all folks!")
