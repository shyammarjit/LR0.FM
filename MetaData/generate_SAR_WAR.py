import os 
import pandas as pd 
import numpy as np 
from collections import defaultdict
import sys
from math import exp
csvs = [e for e in os.listdir('Results/') if ".csv" in e]

resolution  = int(sys.argv[1])
assert resolution in [16, 32, 64, 128, 224], "Only supported resolution are [16, 32, 64, 128, 224]"
    

def fill_in_missing_datasets(row):
    global last_countered
    if row == "":
        return last_countered
    else:
        row = row.lower()  
        if "caltech101" in row:
            row = 'caltech101'
        elif "dtd" in row:
            row = 'dtd'
        elif "food101" in row:
            row = 'food101'
        elif "stanford" in row:
            row = 'stanford_cars'
        elif "fgvc" in row:
            row = 'fgvc_aircraft'
        elif "pets" in row:
            row = 'oxford_pets'
        row = row.replace("\n", " ").replace("\r", " ").replace(" ", "")
        last_countered = row
        return row


def fix_resolution(row):
    if "(original)" in str(row):
        row = row.replace("(original)", "")
    return float(row)
    

Datasets_names = {'imagenet-r', 'imagenet-v2', 'stanford_cars', 'flowers102', 'food101', 'imagenet-a', 'fgvc_aircraft', 'imagenet-sketch', 'imagenet', 'sun397', 'caltech101', 'ucf101', 'eurosat', 'oxford_pets', 'dtd'}
Datasets_scores = defaultdict(list)


WAR_weights = {
    'imagenet': 0.15556157429688613, 'imagenet-a': 0.970498446080589, 
    'imagenet-v2': 0.2854574367981364, 'imagenet-r': 0.01, 
    'imagenet-sketch': 0.021456095637452655, 'caltech101': 0.01, 'dtd': 0.505922498560715, 
    'food101': 0.01, 'sun397': 0.407563119725743, 'stanford_cars': 0.13583821249199218, 
    'fgvc_aircraft': 0.8229545014750042, 'oxford_pets': 0.08995285864599148, 'flowers102': 0.08972060770047119, 
    'eurosat': 1.0, 'ucf101': 0.01
}

overall_dataset_wts = sum([WAR_weights[e] for e in WAR_weights])

classes_per_dataset= {
    'imagenet': 1000, 'imagenet-a': 200,
    'imagenet-v2': 1000, 'imagenet-r': 200, 
    'imagenet-sketch': 1000, 'caltech101': 102, 'dtd': 47, 
    'food101': 102, 'sun397': 397, 'stanford_cars': 196, 
    'fgvc_aircraft': 102, 'oxford_pets': 37, 'flowers102': 102, 
    'eurosat': 10, 'ucf101': 101
}

exp = exp(1)
robust_calc = lambda x: 1 - exp ** (-200 * x ** 2)



Models = []    

category = '_top1'  # ['_top1', '_top5']
for csv in csvs:

    Model_names = []
    last_countered = None
    df = pd.read_csv( os.path.join('Results/', csv))
    # print(csv, df)
    df['Dataset'] = df['Dataset'].fillna('')
    df['Dataset'] = df['Dataset'].apply(fill_in_missing_datasets)
    df['Dataset'].unique()

    pre_column = None 
    new_columns = {}
    for old_col in df.columns:
        new_col = old_col.replace("\n", " ").replace("\r", " ").replace("  ", " ")
        if 'Unnamed' in new_col:
            new_col = pre_column + "_top5" 
        else:
            if ("Dataset" not in  new_col) and ('Image Resolution' not in new_col):
                Model_names.append(new_col)
                pre_column = new_col
                new_col = new_col + "_top1" 
        new_columns[old_col] =  new_col
        
    df = df.rename(columns=new_columns)
    res_column = 'Image Resolution'
    df['Image Resolution'] = df['Image Resolution'].apply(fix_resolution)

    selected = df 
    # selected = df[(df[res_column] == '224 (original)') | (df[res_column] == resolution) | (df[res_column] == str(resolution)) | (df[res_column] == 224)  ]
    # print(selected)
    
    for model in Model_names:
        Models.append(model)
        selected_model = selected[['Dataset', 'Image Resolution', model+ "_top1", model+ "_top5"]]
        SAR = 0
        WAR = 0 
        for dataset in Datasets_names:
            no_of_classes = classes_per_dataset[dataset]
            
            dataset_scores = selected_model[selected_model.Dataset == dataset]
            hq_score = dataset_scores[dataset_scores['Image Resolution'] == 224][model+ category].item()
            hq_score = float(hq_score)
            accuracy_gap = hq_score / 100 - 1/  no_of_classes

            
            lq_score = dataset_scores[dataset_scores['Image Resolution'] == resolution][model+ category].item()
            lq_score = float(lq_score)

            relative_robustness = abs(lq_score  / hq_score)
            robustness_improved = abs(relative_robustness * robust_calc(accuracy_gap))

            # print(relative_robustness, robustness_improved)
            SAR += abs(relative_robustness ) / len(Datasets_names)
            war_score = abs(robustness_improved * WAR_weights[dataset] ) / overall_dataset_wts
            WAR += war_score 
            

            Datasets_scores[dataset + "_Relative-Robustness"].append(relative_robustness) 
            Datasets_scores[dataset + "_Improved-Relative-Robustness"].append(robustness_improved) 

        # print(WAR)
        Datasets_scores["SAR_Relative-Robustness"].append(SAR) 
        Datasets_scores["WAR_Improved-Relative-Robustness"].append(WAR)
        


Datasets_scores['Model'] = Models
df = pd.DataFrame(Datasets_scores)
df = df.set_index('Model')

df = df.sort_values(by=['WAR_Improved-Relative-Robustness'], ascending=False)
df.to_csv(f'WAR_SAR_Ranking/ALL_{resolution}.csv', index=True)


print("Effectiveness of Improved Robsutness as shown in Table 9 of Supplementary")

Aircraft = df[['fgvc_aircraft_Relative-Robustness', 'fgvc_aircraft_Improved-Relative-Robustness']]
print(Aircraft.loc[['ALBEF (4M)', 'ALBEF (14M)', 'ALBEF  (14M + coco_finetuned)', 'ALBEF  (14M + flickr_finetuned)']])
print(Aircraft.loc[['BLIP-ViT-B/16 (4M)', 'BLIP-ViT-B/16 (129M)', 'BLIP-ViT-B/16 & CapFilt-L (129M)', 'BLIP-ViT-L/16 (129M)', 'BLIP-ViT-B/16  (129M + COCO)', 'BLIP-ViT-B/16  (129M + Flickr)', 'BLIP-ViT-L/16  (129M + COCO)', 'BLIP-ViT-L/16  (129M + Flickr)']])

EuroSAT = df[['eurosat_Relative-Robustness', 'eurosat_Improved-Relative-Robustness']]
print(EuroSAT.loc[['ALBEF (4M)', 'ALBEF (14M)']])

CARS = df[['stanford_cars_Relative-Robustness', 'stanford_cars_Improved-Relative-Robustness', ]]
print(CARS.loc[['ALBEF (4M)']])
 
   

print("SAR AND WAR SCORES in Table 2:")
RANKS = df[['SAR_Relative-Robustness', 'WAR_Improved-Relative-Robustness']]
print(RANKS.loc[['EVA-02-CLIP-B/16', 'MetaCLIP- ViT-B/16 (2.5B)', 'OpenCLIP-ViT-B/16']])







# python generate_SAR_WAR.py 16 


