import utils

ALL_CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz' , 'O2']
#ALL_MODELS = ['K-NN', 'SVM', 'DTC', 'RFC', 'LR', 'NN', 'NN-def', 'SVM-def', 'XGB']
ALL_MODELS = ['K-NN', 'SVM', 'DTC', 'RFC', 'LR', 'NN', 'XGB']
SELECTED_CHANNELS = ['CP3', 'Cz', 'CPz', 'P3']
MAIN_CSV_FILE = 'data/final_mels.csv'

dataset = utils.data_loader(MAIN_CSV_FILE)
reduced_dataset = utils.channel_selection(dataset, SELECTED_CHANNELS)
ALL_FEATURES = reduced_dataset.columns[:-2]