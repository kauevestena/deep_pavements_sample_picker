from quality_funcs import *

chosen_samples_metadata = chosen_samples_metadata.drop_duplicates()

# iterate through the reports:
for raw_report_name in tqdm(os.listdir(default_res_path)):
    try:
        all_res_paths = [os.path.join(level_path,raw_report_name) for level_path in resolution_paths]

        all_dfs = [ pd.read_csv(res_path).set_index('model').T*100 for res_path in all_res_paths]

        category = raw_report_name.split('_')[0]
        img_id = raw_report_name.split('_')[1]
        clip_id = raw_report_name.split('_')[2].split('.')[0]

        report_basename = raw_report_name.split('.')[0]

        sample_id = f'{img_id}_{clip_id}'


        # find rows in "chosen_samples_metadata" that match the img_id:
        img_path, sample_path = get_img_paths(sample_id,category)

        if img_path and sample_path:
            write_html_report(report_basename,img_path,sample_path,all_dfs)
            
            img_path = adapth_path(img_path)
            sample_path = adapth_path(sample_path)

            # writing a markdown report both images and all the tables on it:
            write_markdown_report(report_basename,img_path,sample_path,all_dfs)
    except Exception as e:
        print('error:',raw_report_name,str(e))

