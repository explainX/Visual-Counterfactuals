from flask import Flask, session, url_for, render_template
from flask import request, jsonify, json, redirect
from flask import render_template
import pandas as pd
import numpy as np

from explainx_cf.WebApplication.model import *
from explainx_cf.WebApplication.utils import *
from explainx_cf.WebApplication.individual_explanation import *
from explainx_cf.WebApplication.global_explanations import *
from explainx_cf.WebApplication.queries import *
from explainx_cf.WebApplication.d3_functions import *
from explainx_cf.WebApplication.preprocessing import create_summary_file
from explainx_cf.WebApplication.distance_function import generate_projection_files, reduce_raw_data
from explainx_cf.WebApplication.projection import show_projection2, full_projection
import pyrebase
from explainx_cf.WebApplication.config_det import data_det
from functools import wraps


# ------- Initialize WebApp ------- #


def data_in(d):
    global PD
    PD = d
    global data_name
    global lock
    global folder_path
    global data_path
    global preproc_path
    global projection_changes_path
    global projection_anchs_path
    global no_bins
    global df
    global model_path
    global density_fineness
    global categorical_cols
    global monotonicity_arr
    global feature_selector_input
    global percentage_filter_input
    global feature_names
    global all_data
    global data
    global metadata
    global target
    global no_samples
    global no_features
    global svm_model
    global bins_centred
    global X_pos_array
    global init_vals
    global col_ranges
    global all_den
    global all_median
    global all_mean
    global dict_array
    global dict_array_orig
    global reduced_data_path
    global bins_used

    bins_used = PD['bins_used']
    reduced_data_path = PD['reduced_data_path']
    data_name = PD['data_name']
    lock = PD['lock']
    folder_path = PD['folder_path']
    data_path = PD['data_path']
    preproc_path = PD['preproc_path']
    projection_changes_path = PD['projection_changes_path']
    projection_anchs_path = PD['projection_anchs_path']
    no_bins = PD['no_bins']
    df = PD['df']
    model_path = PD['model_path']
    density_fineness = PD['density_fineness']
    categorical_cols = PD['categorical_cols']
    monotonicity_arr = PD['monotonicity_arr']
    feature_selector_input = PD['feature_selector_input']
    percentage_filter_input = PD['percentage_filter_input']
    feature_names = PD['feature_names']
    all_data = PD['all_data']
    data = PD['data']
    metadata = PD['metadata']
    target = PD['target']
    no_samples = PD['no_samples']
    no_features = PD['no_features']
    svm_model = PD['svm_model']
    bins_centred = PD['bins_centred']
    X_pos_array = PD['X_pos_array']
    init_vals = PD['init_vals']
    col_ranges = PD['col_ranges']
    all_den = PD['all_den']
    all_median = PD['all_median']
    all_mean = PD['all_mean']
    dict_array = PD['dict_array']
    dict_array_orig = PD['dict_array_orig']


app = Flask(__name__)
firebase_app = pyrebase.initialize_app(data_det)
session_global = dict()


def check_token(f):
    @wraps(f)
    def wrap(*args, **kwargs):

        try:
            if session_global['authentication']:
                request.user = session_global['authentication']
        except Exception as e:

            return redirect(url_for("main"))
        return f(*args, **kwargs)

    return wrap


@app.route('/paper_intro')
@check_token
def paper_intro():
    print("inside paper_intro")
    return render_template("index_intro_paper.html")


@app.route("/", methods=['POST', 'GET'])
def main():
    return redirect("/login_account")


@app.route('/login_account')
def login_account():
    return render_template("login.html")


@app.route('/challenge_intro')
@check_token
def intro_site_challenge():
    return redirect("http://nyuvis-web.poly.edu/projects/fico/intro")


# @app.route('/paper_intro', methods=['POST', 'GET'])
# @check_token
# def intro_site_paper():
#     return render_template("index_intro_paper.html")


@app.route("/login", methods=['POST', 'GET'])
def login():
    email = request.form.get('email')
    password = request.form.get('psw')
    try:
        user = firebase_app.auth().sign_in_with_email_and_password(email, password)
        jwt = user['idToken']
        session_global['authentication'] = jwt
        if jwt:
            print("jwt")
            return redirect(url_for("paper_intro"))

    except Exception as e:
        return redirect(url_for("main"))


@app.route('/change_dataset', methods=['GET'])
def handle_change_dataset():
    if request.method == 'GET':
        dataset_name = request.args.get('dataset')
        print(dataset_name)

        global Pd
        Pd = PD

        return PD["preproc_path"]


# ------- Individual Explanations ------- #

@app.route('/individual')
@check_token
def ind_site():
    return render_template("index_individual.html", no_samples=no_samples, no_features=no_features,
                           preproc_path=preproc_path[27:], locked=lock)


@app.route('/instance', methods=['GET'])
def handle_request():
    np.random.seed(0)

    if request.method == 'GET':
        sample = -10
        try:
            sample = int(request.args.get('sample'))
        except:
            return "Please enter a sample number in the range (1, " + str(no_samples) + ")."

        if sample != -10:
            if sample < 1 or sample > no_samples:
                return "Please enter a sample number in the range (1, " + str(no_samples) + ")."
            else:
                sample -= 1
                row = data[sample]

                monot = (request.args.get('monot') == "True")
                sort = (request.args.get('sort') == "True")
                density = request.args.get('density')
                lock = request.args.get('locked_features')[1:-1].split(',')

                if lock[0] == "none":
                    lock = []
                if type(lock) in [int, str]:
                    lock = [lock]
                lock = [int(e) for e in lock]
                # print(lock)

                sample, good_percent, model_correct, category, predicted = display_data(data, target, svm_model, sample,
                                                                                        row)

                ### Run MSC and Anchors
                # print(lock)
                change_vector, change_row, anchors, percent = instance_explanation(svm_model, data, row, sample,
                                                                                   X_pos_array,
                                                                                   bins_centred, no_bins,
                                                                                   monotonicity_arr, col_ranges, 1,
                                                                                   True, lock)

                ### Parse values into python dictionary
                data_array = prepare_for_D3(row, bins_centred, change_row, change_vector, anchors, percent,
                                            feature_names, monot, monotonicity_arr, lock)
                dens_array = []
                if density == "High":
                    dens_array = high_den
                elif density == "Low":
                    dens_array = low_den
                else:
                    dens_array = all_den

                if sort:
                    data_array, dens_array = sort_by_val(data_array, dens_array)

                ret_arr = [data_array, dens_array]

                # text_exp = generate_text_explanation(good_percent, X[sample], change_row, change_vector , anchors)
                # similar_ids = detect_similarities("static/data/pred_data_x.csv","static/data/final_data_file.csv", data[sample], change_row, bins_centred, good_percent)
                # similar_ids = similar_ids[:min(len(similar_ids),10)]
                ret_arr.append({'sample': np.float64(sample + 1), 'good_percent': np.float64(good_percent),
                                'model_correct': model_correct,
                                'category': category, 'predicted': np.float64(predicted)})

                return json.dumps(ret_arr)


# ------- New Projection ------- #

@app.route('/projection')
@check_token
def projection_site():
    return render_template("index_projection.html", no_features=no_features,
                           feature_names=json.dumps(feature_names.tolist()), preproc_path=preproc_path,
                           feature_selector_input=json.dumps(feature_selector_input),
                           percentage_filter_input=json.dumps(percentage_filter_input))


@app.route('/main_backend_req', methods=['GET'])
def main_site_backend_req():
    if request.method == 'GET':

        doing_comparison = int(request.args.get('doing_comparison'))
        ft_list = request.args.get('selected_fts')
        ft_list = ft_list[1:-1].split(',')

        if ft_list[0] == '-1' or ft_list == ['']:
            ret_arr = list(range(data.shape[0]))

        else:
            ft_list = [int(x) for x in ft_list]
            ft_list.sort()
            ret_arr = ids_with_combination(preproc_path, ft_list, anchs=False)

        confusion_mat_1 = request.args.get('confusion_mat_1').split(',')
        pred_range_1 = request.args.get('pred_range_1').split(',')
        pred_range_1 = [int(x) for x in pred_range_1]
        modified_range_idx_1 = request.args.get('modified_range_idx_1').split(',')

        if modified_range_idx_1[0] != '':
            modified_range_idx_1 = [int(x) for x in modified_range_idx_1]
        else:
            modified_range_idx_1 = []
        print("MODIFIED RANGE INDEXES", modified_range_idx_1)

        ft_curr_range_1 = request.args.get('ft_curr_range_1').split(',')
        ft_curr_range_1 = [int(x) for x in ft_curr_range_1]
        temp_curr_range = []
        idx = 0
        while idx <= len(ft_curr_range_1) - 2:
            temp_curr_range.append((ft_curr_range_1[idx], ft_curr_range_1[idx + 1]))
            idx += 2
        ft_curr_range_1 = temp_curr_range
        print("FT CURR RANGES", ft_curr_range_1)

        if doing_comparison:
            confusion_mat_2 = request.args.get('confusion_mat_2').split(',')
            pred_range_2 = request.args.get('pred_range_2').split(',')
            pred_range_2 = [int(x) for x in pred_range_2]
            modified_range_idx_2 = request.args.get('modified_range_idx_2').split(',')

            if modified_range_idx_2[0] != '':
                modified_range_idx_2 = [int(x) for x in modified_range_idx_2]
            else:
                modified_range_idx_2 = []
            print("MODIFIED RANGE INDEXES CMP", modified_range_idx_2)

            ft_curr_range_2 = request.args.get('ft_curr_range_2').split(',')
            ft_curr_range_2 = [int(x) for x in ft_curr_range_2]
            temp_curr_range = []
            idx = 0
            while idx <= len(ft_curr_range_2) - 2:
                temp_curr_range.append((ft_curr_range_2[idx], ft_curr_range_2[idx + 1]))
                idx += 2
            ft_curr_range_2 = temp_curr_range
            print("FT CURR RANGE CMP", ft_curr_range_2)

        # STEFFEN
        all_points = full_projection(reduced_data_path + "_" + "PCA" + ".csv", preproc_path)  # REMOVE SOON
        all_samples = [x for x in range(no_samples)]

        idx_range = range(2) if doing_comparison else range(1)

        confusion_mat = confusion_mat_1
        pred_range = pred_range_1
        pred_range = pred_range_1
        modified_range_idx = modified_range_idx_1
        ft_curr_range = ft_curr_range_1
        ret_string = ""

        for idx_k in idx_range:

            # Note: filter_dict always generates the filter_lst so always manipulate the filter_dict variable

            # ==== Filter Dictionary ====
            # Note: order is TP, FP, FN, TN
            conf_list = [0, 0, 0, 0]
            for label in confusion_mat:
                if label == "TP":
                    conf_list[0] = 1
                if label == "FP":
                    conf_list[1] = 1
                if label == "FN":
                    conf_list[2] = 1
                if label == "TN":
                    conf_list[3] = 1

            # // Filter Legend:
            # // 1 - Model Accuracy Range
            # // 2 - Prediction Label
            # // 3 - Feature RangeS

            filter_dict = {"1": [pred_range[0], pred_range[1]], "2": conf_list, "3": []}

            for c in range(len(col_ranges)):
                # [low, high, changed]
                if (c in modified_range_idx):
                    filter_dict["3"].append([ft_curr_range[c][0], ft_curr_range[c][1], 1])
                else:
                    filter_dict["3"].append([ft_curr_range[c][0], ft_curr_range[c][1], 0])

            # ==== Filter Dictionary -> D3 ====
            filter_lst = []
            # Note: this logic might reorder the filters (future fix)

            # --- Model Percentage ---
            if ((pred_range[0] != 0) or (pred_range[1] != 100)):
                single_filter = [1, {"low": pred_range[0], "high": pred_range[1]}]
                filter_lst.append(single_filter)

            # --- Confusion Matrix ---
            if (not all(v == 1 for v in conf_list)):
                single_filter = [2, {"tp": conf_list[0], "fp": conf_list[1], "fn": conf_list[2], "tn": conf_list[3]}]
                filter_lst.append(single_filter)

            # --- Feature Selector ---
            for f in range(len(filter_dict["3"])):
                feat = filter_dict["3"][f]
                if (feat[2] == 1):
                    single_filter = [3, {"name": feature_names[f], "low": feat[0], "high": feat[1]}]
                    filter_lst.append(single_filter)

            # print("FILTER LIST", filter_lst) # This needs to go to D3 filter selector as input

            # === Apply Masks ===

            start_mask = np.ones(data.shape[0])

            mask1 = query_pred_range(metadata, pred_range)
            mask2 = query_confusion_mat(metadata, confusion_mat)
            # mask3 = query_feature_combs(metadata, [15,19])
            # mask4 = query_value_range(data, 0, 60, 70)
            # mask5 = query_similar_points(data,metadata,10,0.5)
            # mask6 = query_sampled_data(data, 30)

            mask4 = np.copy(start_mask)
            for idx in modified_range_idx:
                mask4 = mask4 * query_value_range(data, idx, ft_curr_range[idx][0], ft_curr_range[idx][1])

            current_mask = start_mask * mask1 * mask2 * mask4  # *mask6

            # -- CAN BE REMOVED WHEN FILTER SUMMARY REMOVED --
            result = apply_mask(all_points, current_mask)
            summary = prep_filter_summary(result, no_samples)

            selected_samples = apply_mask(all_samples, current_mask)

            # STEFFEN
            print("SELECTED SAMPLES", selected_samples)
            updated_percentage_filter_input = prep_percentage_filter(metadata, bins_used, selected_samples)
            # updated_conf_matrix_input = prep_confusion_matrix(metadata, selected_samples) # STEFFEN: Need to check if same as prep_summary
            print("PERC FILTER INPUT", updated_percentage_filter_input)
            # print("CONF MAT INPUT", updated_conf_matrix_input)

            ## Parse values into python dictionary
            jsmask1 = [0]
            jsmask2 = current_mask.tolist()
            jsmask1.extend(jsmask2)

            if idx_k == 0:
                ret_string = [result, jsmask1, summary, filter_lst, "null", "null", "null",
                              updated_percentage_filter_input, "null"]
                if doing_comparison:
                    confusion_mat = confusion_mat_2
                    pred_range = pred_range_2
                    pred_range = pred_range_2
                    modified_range_idx = modified_range_idx_2
                    ft_curr_range = ft_curr_range_2
            else:
                ret_string[4] = jsmask1
                ret_string[5] = summary
                ret_string[6] = filter_lst
                ret_string[8] = updated_percentage_filter_input

        return json.dumps(ret_string)


# OSCAR: integrate in main backend call
@app.route('/violin_req')
def violin_site_req():
    if request.method == 'GET':

        proj_samples_1 = request.args.get('id_list_1').split(',')
        proj_samples_2 = request.args.get('id_list_2').split(',')
        sort_val = request.args.get('sort_val')

        if (proj_samples_1[0] == '' or proj_samples_1[0] == '-1'):
            return "-1"

        else:
            # --- Sort Features --- OSCAR!
            sort_toggle = False
            if sort_val != "Default": sort_toggle = True
            print("SORTING IS", sort_toggle)

            doing_comparison = 1 if proj_samples_2[0] != "null" else 0

            # --- Converting to 0 indexed ---
            proj_samples_1 = np.array([int(x) - 1 for x in proj_samples_1])  # [:sample_cap])
            if doing_comparison:
                proj_samples_2 = np.array([int(x) - 1 for x in proj_samples_2])  # [:sample_cap])

            # --- Potential logic for scaling up ---
            if doing_comparison:
                no_comparisons = 2
                samples_lst = [proj_samples_1, proj_samples_2]

            if not doing_comparison:
                no_comparisons = 1
                samples_lst = [proj_samples_1]

            complete_data = []

            for i in range(no_comparisons):
                samples = samples_lst[i]

                # --- Complete data prep for D3 ---
                one_set = prep_complete_data(metadata, data, feature_names, samples, col_ranges, bins_centred,
                                             X_pos_array, 0)
                complete_data.append(one_set)

            # ==== Sorting options ===
            if sort_toggle and no_comparisons > 1:
                # --- Median sort ---
                sort_lst = sort_by_med(complete_data[0]["median"], complete_data[1]["median"])

                complete_data = apply_sort(sort_lst, complete_data)

            ret_string = json.dumps(complete_data)

            return ret_string


@app.route('/table_req')
def table_site_req():
    if request.method == 'GET':

        table_samples = request.args.get('id_list').split(',')

        if (table_samples[0] == '' or table_samples[0] == '-1'):
            return "-1"

        else:
            table_samples = np.array([int(x) - 1 for x in table_samples])  # [:sample_cap])
            ret_string = json.dumps([metadata[table_samples, :3].tolist(), data[table_samples].tolist()])
            # print(ret_string)
            # ret_string = json.dumps([aggr_data, all_den, select_den, all_median , select_median])
            return ret_string


# ------- Run WebApp ------- #

def run_app():
    np.random.seed(12345)
    app.run(port=8880)
