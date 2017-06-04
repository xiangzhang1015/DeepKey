# DeepKey
Python code of DeepKey system 
In the DeepKey experiment, two kinds of data are used: the gait and the EEG data.

1. walking_all.mat in matlabwork folder is the 8 subjects' gait data. It contains 160,000 sample pairs with each pair has 51 features and 1 label.

2. EEG_ID_label6.mat is the EEG data with 8 person and each 1350 samples.

3. EEG/Gait_recognize.py is the recognize code for EEG and Gait, respectively.

4. Invalid_filter.py is the invalid filter code, it using the AR_ID_8person.mat data is the normlized version of walking_all.mat. 
