# Dataset

This repository contains training dataset for job failure prediction based on their resource usages. The data columns are as follows:
```
job_id: unique job id
memory_GB: memory usage in GB
network_log10_MBps: network usage rate in MBps in log scale
local_IO_log10_MBps: total (read+write) local I/O rate in MBps in log scale
NFS_IO_log10_MBps: total parallel file system I/O rate in MBps in log scale
failed: 1 -> failed, 0 -> successful
```

# Problem
Train an ML model to predict whether a job is going to fail or not given their current resource usages.

# Evaluation 

Maximize balanced accuracy: balanced accuracy is defined as the average accuracy of positive and negative classes. Negative class means job is successful, positive class means job has failed. 

Here is a sample code to calculate balanced accuracy in Python.
```
from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_ground_truth, y_model_prediction)
pos_class_acc = CM[1,1]/(CM[1,0]+CM[1,1])
neg_class_acc = CM[0,0]/(CM[0,0]+CM[0,1])
print("Accuracy of positive class:",pos_class_acc)
print("Accuracy of negative class:",neg_class_acc)
print('Balanced accuracy', (pos_class_acc+neg_class_acc)/2)
```
