unzip toydesk_data_full.zip 

# put the center.txt in the corresponding folder. This file contains the scene center and scale information
cp desk2/*.txt toydesk_data/processed/our_desk_2
# put the label_mapping.txt in the corresponding folder. This file contains semantic label mapping index
cp desk1/*txt toydesk_data/processed/our_desk_1

# cp the train/test split file
cp toydesk_data/split/our_desk_2_train_0.8/*.txt toydesk_data/processed/our_desk_2
cp toydesk_data/split/our_desk_1_train_0.8/*.txt toydesk_data/processed/our_desk_1