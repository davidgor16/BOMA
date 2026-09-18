# Archived research utility. Original behavior and file paths are preserved.
import sys
lines = open('/home/tfe/tfg/davidfg/trans/M3C_REVISTA/gopt/src/traintest.py').readlines()
with open('/home/tfe/tfg/davidfg/trans/M3C_REVISTA/hmamba/dataset.py', 'w') as f:
    f.write('import torch\nimport numpy as np\nimport json\nfrom torch.utils.data import Dataset\n\n')
    for i in range(511, 560):
        if 'def __init__' in lines[i]:
            f.write(lines[i].replace('def __init__(self, set, am=\'librispeech\'):', 'def __init__(self, set, am=\'librispeech\', **kwargs):'))
        else:
            f.write(lines[i])
