import sys
sys.path.append('.')
from libs.mapillary_funcs import *

n_samples = 2500

voc1 = ['road','sidewalk']
voc2 = ['asphalt','concrete','grass','ground','sett','paving stones','raw cobblestone','gravel','sand']
voc3 = [f'{j} {i}' for i in voc1 for j in voc2]

voc_list = [voc1,voc2,voc3]

all_vocabs = voc1 + voc2 + voc3

report_header = 'model,' + ','.join(all_vocabs) + '\n'

chosen_samples_path = os.path.join(REPORTS_PATH,'chosen_samples.csv')

def get_chosen_samples_metadata():
    if os.path.exists(chosen_samples_path):
        return pd.read_csv(chosen_samples_path)

def write_to_raw_report(report_name,content):
    report_path = os.path.join(raw_reports_path,report_name+'.csv')

    if not os.path.exists(report_path):
        append_to_file(report_header,report_path)

    append_to_file(content,report_path)



raw_reports_path = os.path.join(REPORTS_PATH,'raw_reports')


create_dir_if_not_exists(raw_reports_path)