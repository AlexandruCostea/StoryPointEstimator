import json
import numpy as np

def serialize_metrics(mode, metrics):
    if mode == 'train':
        json_str = '{\n'
        json_str += '\t"training_metrics": {\n'
        json_str += f'\t\t"accuracy": {metrics[0]},\n'
        json_str += f'\t\t"macro_f1": {metrics[1]},\n'
        json_str += f'\t\t"weighted_f1": {metrics[2]},\n'
        json_str += f'\t\t"precision": {metrics[3]},\n'
        json_str += f'\t\t"recall": {metrics[4]},\n'
        json_str += '\t\t"confusion_matrix": ' + json.dumps(metrics[5].tolist()) + '\n'
        json_str += '\t}\n'
        json_str += '}'
    elif mode == 'val':
        json_str = '{\n'
        json_str += '\t"validation_metrics": {\n'
        json_str += f'\t\t"accuracy": {metrics[0]},\n'
        json_str += f'\t\t"macro_f1": {metrics[1]},\n'
        json_str += f'\t\t"weighted_f1": {metrics[2]},\n'
        json_str += f'\t\t"precision": {metrics[3]},\n'
        json_str += f'\t\t"recall": {metrics[4]},\n'
        json_str += '\t\t"confusion_matrix": ' + json.dumps(metrics[5].tolist()) + '\n'
        json_str += '\t}\n'
        json_str += '}'
    elif mode == 'both':
        json_str = '{\n'
        json_str += '\t"training_metrics": {\n'
        json_str += f'\t\t"accuracy": {metrics[0]},\n'
        json_str += f'\t\t"macro_f1": {metrics[1]},\n'
        json_str += f'\t\t"weighted_f1": {metrics[2]},\n'
        json_str += f'\t\t"precision": {metrics[3]},\n'
        json_str += f'\t\t"recall": {metrics[4]},\n'
        json_str += '\t\t"confusion_matrix": ' + json.dumps(metrics[5].tolist()) + '\n'
        json_str += '\t},\n'
        json_str += '\t"validation_metrics": {\n'
        json_str += f'\t\t"accuracy": {metrics[6]},\n'
        json_str += f'\t\t"macro_f1": {metrics[7]},\n'
        json_str += f'\t\t"weighted_f1": {metrics[8]},\n'
        json_str += f'\t\t"precision": {metrics[9]},\n'
        json_str += f'\t\t"recall": {metrics[10]},\n'
        json_str += '\t\t"confusion_matrix": ' + json.dumps(metrics[11].tolist()) + '\n'
        json_str += '\t}\n'
        json_str += '}'
    
    return json_str