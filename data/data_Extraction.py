import pandas as pd
import os
import numpy as np
import shutil
import requests

df = pd.read_csv(r"D:\AI\projects\deep fake detection\data\FINAL_DATASET.csv")


valid_rows = []

skipped = 0

for index , rows in df.iterrows():

    try:

        url = df['image_url'].iloc[index]
        label = df['label_numeric'].iloc[index]
        dataset_split = df['dataset_split'].iloc[index]

        print(url)

        image = requests.get(url, timeout=10)

        if image.status_code != 200:
            skipped += 1
            continue

        filename = f"{index}.jpg"

        with open(f"images/{filename}", "wb") as f:
            f.write(image.content)

        valid_rows.append({
            "image_name": filename,
            "label": label,
            "dataset_split": dataset_split
        })

    except requests.exceptions.ConnectionError:
        print(f"Connection Error at: {url}")
        skipped += 1
        continue

    except requests.exceptions.Timeout:
        print(f"Timeout at: {url}")
        skipped += 1
        continue

    except Exception as e:
        print(f"Error at {url}: {e}")
        skipped += 1
        continue        



image_data = pd.DataFrame(columns = ['image' , 'label'])

for i , data  in enumerate(valid_rows):
    image_data.loc[i , 'image'] = valid_rows[i]['image_name']
    image_data.loc[i , 'label'] = valid_rows[i]['label']
    image_data.loc[i , 'dataset_split'] = valid_rows[i]['dataset_split']



image_data.to_csv('image_label.csv')

image_data['image'] =  r"path of folder of images (All)" + image_data['image'] 

destination_folder_train_AI = r"path of folder of train data of ai"
destination_folder_train_real = r"path of folder of train data of real"
destination_folder_test_AI = r"path of folder of test data of ai"
destination_folder_test_real = r"path of folder of test data of real"
destination_folder_val_AI = r"path of folder of val data of ai"
destination_folder_val_real = r"path of folder of val data of real"


for  i , image_path in enumerate(image_data["image"]):

    if image_data.loc[i ,'dataset_split'] == 'train':
        if image_data.loc[i ,'label'] == 1:
            shutil.copy(image_path, destination_folder_train_real)
        elif image_data.loc[i ,'label'] == 0:
            shutil.copy(image_path, destination_folder_train_AI)

    elif image_data.loc[i ,'dataset_split'] == 'test':
        if image_data.loc[i ,'label'] == 1:
            shutil.copy(image_path, destination_folder_test_real)
        elif image_data.loc[i ,'label'] == 0:
            shutil.copy(image_path, destination_folder_test_AI)

    elif image_data.loc[i ,'dataset_split'] == 'val':
        if image_data.loc[i ,'label'] == 1:
            shutil.copy(image_path, destination_folder_val_real)
        elif image_data.loc[i ,'label'] == 0:
            shutil.copy(image_path, destination_folder_val_AI)
    
