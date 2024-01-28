from quality_funcs import *

data = {
    'id' : [],
    'territory' : [],
    'category' : [],
    'sample_path' : [],
    'img_path' : [],
}

for i in tqdm(range(n_samples)):
    sample = sample_handler()

    
    if sample.check_perspective_camera():

        category = sample.get_random_detection_category()
        sample_path = sample.get_single_clip(category=category)

        if sample_path:
            data['id'].append(sample.mapillary_id)
            data['img_path'].append(sample.img_path)
            data['territory'].append(sample.territory)
            data['category'].append(category)
            data['sample_path'].append(sample_path)


data_df = pd.DataFrame(data)
data_df.to_csv(chosen_samples_path,index=False)

        