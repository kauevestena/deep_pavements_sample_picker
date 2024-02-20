import sys
sys.path.append('.')
from libs.mapillary_funcs import *
import plotly.graph_objects as go
import io
import base64
import shutil
import clip
import open_clip

n_samples = 2500

TEST_REDUCED_SIZE = False



voc_dict = {
    'voc1' : ['road','sidewalk'],
    'voc2' : ['road','sidewalk','other'],

    'voc2b' : [ 'road', 'sidewalk', 'not a pavement'],
    'voc2c' : [ 'road','path','way','track', 'sidewalk', 'not a pavement'],
    'voc2d' : [ 'road','path','way','track'],
    'voc2e' : [ 'road','path','way','track','something else'],
    'voc3' : [ 'road', 'sidewalk', 'not a path'],
    'voc4' : [ 'road', 'sidewalk', 'car', 'tree', 'pole', 'building'],
    'voc4b' : [ 'road', 'sidewalk', 'car', 'tree', 'pole', 'building','other'],

    'voc4c' : ["paved", "unpaved",],
    'voc5' : ["asphalt", "concrete", "cobblestone", "block paving", "unpaved"],
    'voc5b' : ["asphalt", "cement", "cobblestone", "block paving", "unpaved"],
    'voc6' : ['paved','grass','soil','dirt','sand','earth','rocks','gravel','cobblestone'],
    'voc7' : ["asphalt", "concrete", "cobblestone", "block paving", 'gravel', "sett", "grass", "dirt"],
    'voc7b' : ["asphalt", "concrete", "cobblestone", "pavers", "grass", "dirt"],
    'voc8' : ["asphalt", "cement", "cobblestone", "block pavement", "sett pavement", "grass", "dirt"],
    'voc8b' : ["asphalt", "cement", "cobblestone", "block paving", "sett paving", "grass", "dirt"],

    'voc_smooth1' : ['excellent smoothness','good smoothness','average smoothness','poor smoothness'],
    'voc_smooth2' : ['excellent smoothness','good smoothness','average smoothness','bad smoothness','horrible smoothness','very_horrible smoothness','horrible smoothness'],
    'voc_smooth2b' : ['excellent condition','good condition','average condition','bad condition','horrible condition'],
}

voc_names_list = list(voc_dict.keys())

voc_list = [voc_dict[key] for key in voc_dict.keys()]
all_vocabs = [voc for vocs in voc_list for voc in vocs]

report_header = 'model,' + ','.join(all_vocabs) + '\n'

chosen_samples_path = os.path.join(REPORTS_PATH,'chosen_samples.csv')

def reset_raw_reports():
    shutil.rmtree(raw_reports_path,ignore_errors=True)
    create_dir_if_not_exists(raw_reports_path)

def get_chosen_samples_metadata():
    if os.path.exists(chosen_samples_path):
        return pd.read_csv(chosen_samples_path)

if os.path.exists(chosen_samples_path):
    chosen_samples_metadata = get_chosen_samples_metadata()

def write_to_raw_report(report_name,level,content):
    level_folderpath = os.path.join(raw_reports_path,level)

    create_dir_if_not_exists(level_folderpath)

    report_path = os.path.join(level_folderpath,report_name+'.csv')

    if not os.path.exists(report_path):
        append_to_file(report_path,report_header)

    append_to_file(report_path,content)

def write_to_raw_report_v2(report_name,voc_name,vocabulary,content):
    voc_path = os.path.join(raw_reports_path,voc_name)
    create_dir_if_not_exists(voc_path)

    report_path = os.path.join(voc_path,report_name+'.csv')

    header = 'model,' + ','.join(vocabulary) + '\n'

    if not os.path.exists(report_path):
        append_to_file(report_path,header)

    append_to_file(report_path,content)

resolution_levels = ['orig','half','quarter']

default_resolution = 'orig'

raw_reports_path = os.path.join(REPORTS_PATH,'raw_reports')
html_reports_path = os.path.join(REPORTS_PATH,'html_reports')
markdown_reports_path = os.path.join(REPORTS_PATH,'markdown_reports')

default_res_path = os.path.join(raw_reports_path,default_resolution)

sample_voc_path = os.path.join(raw_reports_path,list(voc_dict.keys())[0])

vocs_paths = [os.path.join(raw_reports_path,vocname) for vocname in voc_dict]
resolution_paths = [os.path.join(raw_reports_path,level) for level in resolution_levels]

for dirpath in [raw_reports_path,html_reports_path,markdown_reports_path]:
    create_dir_if_not_exists(dirpath)


def get_img_paths(sample_id,category):

    sample_metadata = chosen_samples_metadata.loc[chosen_samples_metadata['sample_id'] == sample_id ]

    if len (sample_metadata) > 1:
        # check if category matches:
        second_sample = sample_metadata.loc[sample_metadata['category'] == category]

        if len(second_sample) == 1:
            img_path = second_sample['img_path'].iloc[0]
            sample_path = second_sample['sample_path'].iloc[0]
        else:
            return None, None
    else:
        img_path = sample_metadata['img_path'].iloc[0]
        sample_path = sample_metadata['sample_path'].iloc[0]

    if img_path and sample_path:
        return img_path, sample_path
        
    # except Exception as e:
    #     print('error:',sample_id,category,str(e))
    #     return None, None


def adapth_path(path,type = 'outside',system='windows',replacement_dict = {'data':'..'}):
    # ../data/samples/italy/1607355149461157/clipped_detections/earth/1607355149461157_0.png (A SAMPLE PATH)
    # data/samples/italy/1607355149461157/clipped_detections/earth/1607355149461157_0.png (ANOTHER SAMPLE PATH)

    # "outside" means to see outside the container:

    # if system == 'windows':
    #     path = path.replace('/',r'\\')

    if type == 'outside':
        for k,v in replacement_dict.items():
            path = path.replace(k,v)
    else:
        return path
    
    return path

def write_markdown_report(report_basename,img_path,sample_path,all_dfs):
    outpath = os.path.join(markdown_reports_path,report_basename+'.md')

    content = f"""
# {report_basename}

## Original Image:

![original]({img_path})

## Clipped Image:

![clip]({sample_path})

"""

    for ref in zip(resolution_levels,all_dfs):
        content += "\n### " + ref[0] + ':\n\n'
        content += ref[1].to_markdown() + '\n'

    content += '\n'

    write_to_file(outpath,content)



# def write_html_report(report_basename,img_path,sample_path,all_dfs):
#     outpath = os.path.join(html_reports_path,report_basename+'.html')

#     # start content with an html header:
#     content = '<!DOCTYPE html>\n<html>\n<head>\n</head>\n<body>\n'

#     content = f"""
#     <h1>For {report_basename}</h1>

#     <h1>Original Image</h1>
#     <img src={img_path}>

#     <h1>Clip Image</h1>
#     <img src={sample_path}>

#     """

#     for ref in zip(resolution_levels,all_dfs):
#         content += f"""
#         <h1>{ref[0]}</h1>
#         {ref[1].to_html()}
#         """

#     content += '\n'

#     # end contente with a html footer:
#     content += '</body>\n</html>'

#     write_to_file(outpath,content)

def one_digit_float(x):
    return f"{x:.1f}"

def float_to_str(x,places=1):
    return f"{x:.{places}f}"



def gen_rep_df_chart(df,title_text,outpath,full_html=False):
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))

    fig.update_layout(
        title=f'Line Chart for {title_text}',
        # xaxis_title='X-axis Label',
        # yaxis_title='Y-axis Label',
    )

    # with tempfile.tempdir() as tmpdir:
    #     f = os.path.join(tmpdir,f'{title_text}.html')
    return fig.write_html(outpath,full_html=full_html)


def image_to_base64(image_path: str, return_html: bool = False, new_width: int = 600) -> str:
    """
    Read an image, reduce it to the specified pixels width, and transform it to a base64 string or full <img> HTML tag.

    Args:
    image_path: The path to the input image file.
    return_html: If True, return the full <img> HTML tag; if False, return the base64 string. Default is False.
    new_width: The desired width of the image in pixels.

    Returns:
    The base64 representation of the transformed image, or the full <img> HTML tag.
    """
    with Image.open(image_path) as img:
        width_percent = (new_width / float(img.size[0]))
        new_height = int((float(img.size[1]) * float(width_percent)))
        img = img.resize((new_width, new_height))
        img_byte_array = io.BytesIO()
        img.save(img_byte_array, format='JPEG')
        img_byte_array = img_byte_array.getvalue()
        base64_img = base64.b64encode(img_byte_array).decode('utf-8')

        if return_html:
            img_tag = f'<img src="data:image/jpeg;base64,{base64_img}">'
            return img_tag
        else:
            return base64_img

def write_html_report(report_basename, img_path, sample_path, all_dfs,write_tables=False):
    """
    Generate an HTML report with given information.

    Args:
    report_basename (str): The base name for the report.
    img_path (str): The path to the original image.
    sample_path (str): The path to the clipped image.
    all_dfs (list): List of DataFrames to be included in the report.

    Returns:
    None
    """
    # Define the output path for the HTML report
    outpath = os.path.join(html_reports_path, report_basename + '.html')

    # Define the CSS style and the beginning of the HTML content
    content = '''
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    body {
        font-family: 'Open Sans', sans-serif;
    }
    table {
        font-family: 'Open Sans', sans-serif;
        border-collapse: collapse;
        width: 100%;
    }
    th, td {
        border: 1px solid #dddddd;
        text-align: center;
        padding: 8px;
    }
    tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    th {
        background-color: #4CAF50;
        color: white;
    }
    img {
        width: 600px;
    }
    </style>
    </head>
    <body>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"></script>
    '''

    # Add the report title and images to the content
    content += f"""
    <h1>For {report_basename}</h1>

    <h1>Original Image</h1>
    {image_to_base64(img_path, return_html=True)}

    <h1>Clipped Image</h1>
    {image_to_base64(sample_path, return_html=True)}

    """

    # Add the DataFrames to the content
    if write_tables:
        for ref in zip(voc_names_list, all_dfs,strict=True):
            content += f"""
            <h1>{ref[0]}</h1>
            {ref[1].to_html(float_format=float_to_str)}
            """
    
    # chart generation:
    for ref in zip(voc_names_list, all_dfs,strict=True):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, f'temp.html')
            gen_rep_df_chart(ref[1], ref[0], temp_path)
            content += simple_read(temp_path)

    # Complete the HTML content
    content += '''
    </body>
    </html>
    '''

    # Write the content to the output file
    write_to_file(outpath, content)
