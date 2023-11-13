# Gait processing

Gait trials need to be trimmed and re-aligned. The trials are organized per index,
see get_data_info() in data_info.py. For instance, the trial 10mwt from session
dfa1c060-df8d-40e8-90e6-107078621c7c is index 100.


## Trimming

### PARKER
Enter the range in get_data_select_window() in data_info.py.
If the trial looks good, ignore. To use the original start/end, use -1. Most trials
are bad at the end.

Trials to be checked: 0-97. Some are already done, see get_data_select_window()

## Re-alignment

### Scott
Identify trials with calibration (parents) and their children. See how Antoine proceeded
for trials >= 100. The session with calibration is in the comment.

### Antoine
Find angle and re-process files

## Verify data to be tracked
## Verify simulations