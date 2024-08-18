import pandas as pd
import numpy as np
import json
import os
import warnings
from datetime import datetime 
from collections import OrderedDict
from collections import Counter
import plotly.express as px
from collections import defaultdict
import plotly.graph_objects as go

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', None)

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


chunk_size = 1000
file_id = '02'
project_path = "/home/carolus/Documents/school/green_ia/" 
jsonl_02 = project_path + 'data/' + file_id + '_openfoodfacts_02.jsonl' 
jsonl_sample = project_path + 'data/' + file_id + "_openfoodfacts_sample.jsonl"


# récupérer la date du jour 
current_date_time = datetime.now()
date_format = "%d/%m/%Y %H:%M:%S.%f"
start_date = current_date_time.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
date_code = current_date_time.strftime('%d%m%Y%H%M%S') + f"{current_date_time.microsecond // 1000:03d}"


def add_logs(logData):
    print(logData)
    with open(f"{project_path}logs/03_data_analysis_{date_code}_logs.txt", "a") as logFile:
        logFile.write(f'{logData}\n')


add_logs("03_data_analysis logs:")
add_logs(f"chunk_size: {chunk_size} \nfile_id: {file_id} \nproject_path: {project_path} \njsonl_02: {jsonl_02} \njsonl_sample: {jsonl_sample} \nstart_date: {start_date}")


# verifie la validité de la structure du fichier jsonl
with open(jsonl_02, 'r') as file:
    for line in file:
        try:
            json_object = json.loads(line)
        except json.JSONDecodeError as e:
            add_logs(f"ERROR decoding jsonl: {e}")

add_logs(f"jsonl format valid: {jsonl_02}")


# retourne une liste des pays présents dans le fichier
def extract_countries_from_jsonl(file_path):
    countries = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                record = json.loads(line)
                country = record.get('countries')
                if country is not None:
                    countries.append(country)
            except json.JSONDecodeError:
                add_logs(f"WARNING line: {line} in {file_path}")    
    return countries
countries_list = extract_countries_from_jsonl(jsonl_02)

separated_countries = []
for entry in countries_list:
    countries = [country.strip() for country in entry.split(',')]
    separated_countries.extend(countries)

country_counts = Counter(separated_countries)
total_countries = sum(country_counts.values())

data_graph_countries = []
for country, count in country_counts.items():
    percentage = (count / total_countries) * 100
    add_logs(f"{country}: {percentage:.2f}%")
    data_graph_countries.append({'countries': country, 'percentage': percentage})


# retourne une liste des notes écoscores présentes dans le fichier
def extract_grad_ecoscore_from_jsonl(file_path):
    ecoscore_grad_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                record = json.loads(line)
                ecoscore_grad = record.get('ecoscore_note')
                if ecoscore_grad is not None:
                    ecoscore_grad_list.append(ecoscore_grad)
            except json.JSONDecodeError:
                add_logs(f"WARNING line: {line} in {file_path}")
    return ecoscore_grad_list
ecoscore_grad_list = extract_grad_ecoscore_from_jsonl(jsonl_02)

ecoscore_grad_counts = Counter(ecoscore_grad_list)
total_ecoscore_grads = sum(ecoscore_grad_counts.values())
data_graph_ecoscore_grad = []
for ecoscore_grad, count in ecoscore_grad_counts.items():
    percentage = (count / total_ecoscore_grads) * 100
    add_logs(f"{ecoscore_grad}: {percentage:.2f}%")
    data_graph_ecoscore_grad.append({'ecoscore_grad': ecoscore_grad, 'percentage': percentage})


# retourne une liste des lettres écoscore présentes dans le fichier
def extract_groups_ecoscore_from_jsonl(file_path):
    ecoscore_groups_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                record = json.loads(line)
                ecoscore_groups = record.get('ecoscore_groups')
                if ecoscore_groups is not None:
                    ecoscore_groups_list.append(ecoscore_groups)
            except json.JSONDecodeError:
                add_logs(f"WARNING line: {line} in {file_path}")
    return ecoscore_groups_list
ecoscore_groups_list = extract_groups_ecoscore_from_jsonl(jsonl_02)

ecoscore_groups_counts = Counter(ecoscore_groups_list)
total_ecoscore_groups = sum(ecoscore_groups_counts.values())

data_graph_ecoscore_groups = []
for ecoscore_group, count in ecoscore_groups_counts.items():
    percentage = (count / total_ecoscore_groups) * 100
    add_logs(f"{ecoscore_group}: {percentage:.2f}%")
    data_graph_ecoscore_groups.append({'ecoscore_group': ecoscore_group, 'percentage': percentage})


# retourne une liste des labels présents dans le fichier (sans doublons dans l'affichage)
def extract_labels_from_jsonl(file_path):
    labels_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                record = json.loads(line)
                labels = record.get('labels_note')
                if labels is not None:
                    labels_list.append(labels)
            except json.JSONDecodeError:
                add_logs(f"WARNING line: {line} in {file_path}")
    return labels_list

labels_list = extract_labels_from_jsonl(jsonl_02)
label_counts = Counter(labels_list)
total_labels = sum(label_counts.values())

data_graph_labels = []
for label, count in label_counts.items():
    percentage = (count / total_labels) * 100
    add_logs(f"{label}: {percentage:.2f}%")
    data_graph_labels.append({'labels': label, 'percentage': percentage})


def count_none_and_total_values(jsonl_file_path):
    none_counts = defaultdict(int)
    total_counts = defaultdict(int)
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            for key, value in data.items():
                total_counts[key] += 1
                if value is None:
                    none_counts[key] += 1
    return none_counts, total_counts

def calculate_percentage(none_counts, total_counts):
    percentages = {}
    for key in none_counts:
        if total_counts[key] > 0:
            percentage = (none_counts[key] / total_counts[key]) * 100
        else:
            percentage = 0
        percentages[key] = percentage
    return percentages

none_counts, total_counts = count_none_and_total_values(jsonl_02)
percentages = calculate_percentage(none_counts, total_counts)


def count_specific_values(jsonl_file_path):
    counts = {
        'ecoscore_groups': {'z': 0},
        'ecoscore_note': {999: 0}
    }
    total_counts = {
        'ecoscore_groups': 0,
        'ecoscore_note': 0
    }
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            for key, value in data.items():
                if key in counts:
                    total_counts[key] += 1
                    if key == 'ecoscore_groups' and value == 'z':
                        counts[key]['z'] += 1
                    if key == 'ecoscore_note' and value == 999:
                        counts[key][999] += 1
    return counts, total_counts

def calculate_percentage(count, total):
    if total > 0:
        percentage = (count / total) * 100
    else:
        percentage = 0
    return percentage

counts, total_counts = count_specific_values(jsonl_02)

z_percentage = calculate_percentage(counts['ecoscore_groups']['z'], total_counts['ecoscore_groups'])
number_999 = counts['ecoscore_note'][999]
number_999_percentage = calculate_percentage(number_999, total_counts['ecoscore_note'])

labels = ['Ecoscore Groups (z)', 'Ecoscore Note (999)']
values = [z_percentage, number_999_percentage]
counts_values = [counts['ecoscore_groups']['z'], counts['ecoscore_note'][999]]
add_logs(f"ecoscore groups: {z_percentage}")
add_logs(f"ecoscore grad: {number_999_percentage}")


add_logs(f"total product number: {total_labels}")


# récupérer la date du jour 
current_date_time = datetime.now()
end_date = current_date_time.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
add_logs(f"end date: {end_date}")

# afficher temps total execution script 
start_date = datetime.strptime(start_date, date_format)
end_date = datetime.strptime(end_date, date_format)
time_difference = end_date - start_date
time_difference_minutes = time_difference.total_seconds() / 60
add_logs(f"execution script time: {time_difference_minutes:.2f} minutes")