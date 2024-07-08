import torch
from typing import Tuple, List, Dict
import pandas as pd
from copy import deepcopy
from transformers import AutoTokenizer
import json

def get_user_assistant_prefix(model_name) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get user and assistant prefix for different models
    """
    if 'vicuna' in model_name.lower():
        template = ". USER: h ASSISTANT: h</s>" # tokenize as "<s>, _., _US, ER, :, _h, _A, SS, IST, ANT, :, _h, </s>"
        user_prefix_range = range(2, 5) # _US, ER, :
        assistant_prefix_range = range(6, 11) # _A, SS, IST, ANT, :
    elif 'yi' in model_name.lower():
        template = "\n### Human:\n### Assistant:\n" # tokenize as "'▁', '\n', '###', '▁Human', ':', '\n', '###', '▁Assistant', ':', '\n'"
        user_prefix_range = range(2, 5) # '###', '▁Human', ':'
        assistant_prefix_range = range(6, 9) # '###', '▁Assistant', ':'
    elif 'llama-3' in model_name.lower() or 'llama3' in model_name.lower():
        print("identify llama-3")
        template = "<|start_header_id|>user<|end_header_id|>\n\nA<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nB<|eot_id|>" # '<|begin_of_text|>', '<|start_header_id|>', 'user', '<|end_header_id|>', 'ĊĊ', 'A', '<|eot_id|>', '<|start_header_id|>', 'assistant', '<|end_header_id|>', 'ĊĊ', 'B', '<|eot_id|>'
        user_prefix_range = range(1, 5) # '<|start_header_id|>', 'user', '<|end_header_id|>', 'ĊĊ'
        assistant_prefix_range = range(7, 11)
    else:
        raise NotImplementedError
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokens = tokenizer.tokenize(template, add_special_tokens=True)
    token_ids = tokenizer.encode(template, add_special_tokens=True)
    template_tokenized_list = list(zip(tokens, token_ids))

    print("user prefix ", [template_tokenized_list[i] for i in user_prefix_range])
    print("assistant prefix: ", [template_tokenized_list[i] for i in assistant_prefix_range])

    user_prefix = torch.tensor([token_ids[i] for i in user_prefix_range], dtype=torch.long)
    assistant_prefix = torch.tensor([token_ids[i] for i in assistant_prefix_range], dtype=torch.long)
    
    return user_prefix, assistant_prefix


def matrix_name_to_size(matrix_name) -> Tuple[int, int]:
    """
    input: attentionImportance_(10, 10)
    output: 100.0
    """
    return int(matrix_name.split('_')[1].replace('(','').replace(')','').split(',')[0]), int(matrix_name.split('_')[1].replace('(','').replace(')','').split(',')[1])

def generation_result_dict_to_bags(result_dict, uniform_layer, layers) -> Tuple[List, List, List, List]:
    """
    Convert the result dict to bags of balls.
    Inputs:
        result_dict: dict of dict of list, with key as matrix_name, value as method_dict.
            result_dict[matrix_name][method_name] = [accuracy_loss, latency_cost, config_id, config]
    Outputs:
        bags_of_accuracies: list of tensor, the length of list is num_matrices, each tensor with shape (keep_dim, num_balls)
        bags_of_latencies: list of tensor, the length of list is num_matrices, each tensor with shape (keep_dim, num_balls)
        bags_of_config_ids: list of tensor, the length of list is num_matrices, each tensor with shape (keep_dim, num_balls)
        bags_of_method_index: list of tensor, the length of list is num_matrices, each tensor with shape (keep_dim, num_balls)
    """
    # Define Bags
    # bags of choices. num_bags = \sum keep_dim_i * num_plan_config_i
    bags_of_accuracies = [] # list of tensor
    bags_of_latencies = [] # list of tensor
    bags_of_config_ids = [] # list of tensor
    bags_of_method_index = [] # list of tensor

    # move results to bags
    for matrix_name, method_dict in result_dict.items():
        bags_of_accuracies_for_this_matrix = []
        bags_of_latencies_for_this_matrix = []
        bags_of_config_ids_for_this_matrix = []
        bags_of_method_index_for_this_matrix = []

        for method_name, _ in method_dict.items():
            if uniform_layer:
                shape = result_dict[matrix_name][method_name]['config_id'].shape
                accuracy_loss = result_dict[matrix_name][method_name]['accuracy_loss'].reshape(layers, shape[0] // layers, shape[1])
                latency_cost = result_dict[matrix_name][method_name]['latency_cost'].reshape(layers, shape[0] // layers, shape[1])
                bags_of_accuracies_for_this_matrix.append(torch.sum(accuracy_loss, dim=1))
                bags_of_latencies_for_this_matrix.append(torch.sum(latency_cost, dim=1))
            else:
                bags_of_accuracies_for_this_matrix.append(result_dict[matrix_name][method_name]['accuracy_loss'])
                bags_of_latencies_for_this_matrix.append(result_dict[matrix_name][method_name]['latency_cost'])
            bags_of_config_ids_for_this_matrix.append(result_dict[matrix_name][method_name]['config_id'])
            bags_of_method_index_for_this_matrix.append(torch.ones_like(result_dict[matrix_name][method_name]['config_id']) * list(method_dict.keys()).index(method_name))
        
        # concat as a larger bag
        bags_of_accuracies_for_this_matrix = torch.cat(bags_of_accuracies_for_this_matrix, dim=1)
        bags_of_latencies_for_this_matrix = torch.cat(bags_of_latencies_for_this_matrix, dim=1)
        bags_of_config_ids_for_this_matrix = torch.cat(bags_of_config_ids_for_this_matrix, dim=1)
        bags_of_method_index_for_this_matrix = torch.cat(bags_of_method_index_for_this_matrix, dim=1)

        # add as a new bag
        bags_of_accuracies.append(bags_of_accuracies_for_this_matrix)
        bags_of_latencies.append(bags_of_latencies_for_this_matrix)
        bags_of_config_ids.append(bags_of_config_ids_for_this_matrix)
        bags_of_method_index.append(bags_of_method_index_for_this_matrix)
    
    return bags_of_accuracies, bags_of_latencies, bags_of_config_ids, bags_of_method_index

def restore_opt_result(opt_cofig_index_list, sizes):
    '''
    restore the opt result to the original result
    '''
    opt_result = []
    for config, size in zip(opt_cofig_index_list, sizes):
        opt_result.append(config.repeat_interleave(size, dim=0))
    return opt_result


def update_chosen_method_to_result_dict(opt_cofig_index_list, result_dict, bag_sizes):
    """
    Update the chosen method to result_dict.
    Inputs:
        opt_cofig_index_list: list of tensor, the length of list is num_bags, each tensor with shape (keep_dim)
        result_dict: dict of dict of list, with key as matrix_name, value as method_dict.
            result_dict[matrix_name][method_name] = [accuracy_loss, latency_cost, config_id, config]
        bag_sizes: list of int, the length of list is num_bags, each int is the number of balls in this bag
    """

    def _index_list_to_one_hot(opt_cofig_index_list, bag_sizes):
        one_hot_list = []
        for i, opt_cofig_index in enumerate(opt_cofig_index_list):
            num_rows = opt_cofig_index.shape[0]
            one_hot_list.append(torch.zeros(num_rows, bag_sizes[i], dtype=torch.int))
            one_hot_list[i][torch.arange(num_rows), opt_cofig_index] = 1
        return one_hot_list
    
    one_hot_opt_cofig_index_list = _index_list_to_one_hot(opt_cofig_index_list, bag_sizes)
    
    # add one_hot_opt_cofig_index_list to result dict
    for i, (matrix_name, method_dict) in enumerate(result_dict.items()):
        start_index = 0
        for method_name in method_dict.keys():
            shape = result_dict[matrix_name][method_name]['config_id'].shape
            result_dict[matrix_name][method_name]['chosen'] = one_hot_opt_cofig_index_list[i][:, start_index:start_index+shape[1]]
            start_index += shape[1]

def generation_result_dict_to_df(result_dict, weight_name_dict, num_layer, num_head) -> pd.DataFrame:
    """
    Convert the result dict as a dataframe, with
        Columns: matrix_name, method_name, accuracy_loss, latency_cost, config_id, name, chosen
        Rows: different configs
    Inputs:
        result_dict: dict, with key as matrix_name, value as method_dict
        weight_name_dict: dict, with key as matrix_shape_str, value as list of weight names
        num_layer: int, number of attention layers
        num_head: int, number of attention heads
    """
    # Initialize an empty list to store rows
    rows = []
    
    for matrix_name, method_dict in result_dict.items():
        matrix_shape_str = matrix_name.split('_')[-1]
        if matrix_name.split('_')[0] == 'attentionImportance':
            name_dict = {matrix_shape_str: ['model.layers.{}.self_attn.attn_matrix.activation.{}'.format(n, m) for n in range(num_layer) for m in range(num_head)]}
        else:
            name_dict = weight_name_dict
        for method_name, result_dict_values in method_dict.items():
            keep_dim = result_dict_values['config_id'].shape[0]
            num_id = result_dict_values['config_id'].shape[1]
            for dim in range(keep_dim):
                for id in range(num_id):
                    # Create a dictionary for each row and append it to the list
                    # TODO: optimize this matching process
                    row = {
                        'matrix_name': matrix_name,
                        'method_name': method_name,
                        'name': name_dict[matrix_shape_str][dim],
                        'accuracy_loss': result_dict_values['accuracy_loss'][dim][id].item(),
                        'latency_cost': result_dict_values['latency_cost'][dim][id].item(),
                        'config_id': result_dict_values['config_id'][dim][id].item(),

                        'chosen': result_dict_values['chosen'][dim][id].item(),
                        'primary_config': result_dict_values['primary_config'][dim][id].item(),
                    }
                    rows.append(row)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(rows)

    return df


def result_df_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the result of df.
    Inputs:
        df: pandas DataFrame with columns ['matrix_name', 'method_name', 'accuracy_loss', 'latency_cost', 'config_id', 'config', 'chosen']
        config_dict: dict of dict of list, with key as matrix_name, value as method_dict.
            config_dict[matrix_name][method_name] = [primary_config for config_id in config_ids]
    Outputs:
        df: pandas DataFrame with columns ['matrix_name', 'method_name', 'accuracy_loss', 'latency_cost', 'config_id', 'config', 'matrix_size', 'matrix_method', 'accuracy_loss_norm', 'latency_cost_norm', 'latency_loss_multiply_norm', 'layer', 'weight_name']
    """

    df['matrix_size'] = df['matrix_name'].apply(lambda x: matrix_name_to_size(x)[0] * matrix_name_to_size(x)[1])
    df['matrix_method'] = df['matrix_name'] + '_' + df['method_name']
    df['accuracy_loss_norm'] = df['accuracy_loss'] / df['matrix_size']
    df['latency_cost_norm'] = df['latency_cost'] / df['matrix_size']
    df['latency_loss_multiply_norm'] = df['latency_cost_norm'] * df['accuracy_loss_norm']
    df['layer'] = df['name'].apply(lambda x: int(x.split('.')[2]) if type(x) is str else -1) # for weight
    df['weight_name'] = df['name'].apply(lambda x: x.split('.')[4] if type(x) is str else 'attention')

    # if method_name uses something like 4bitQuant_METHOD, then extract the bitwidth and substract it from method_name, only leave METHOD
    df['bitwidth'] = df['method_name'].apply(lambda x: (x.split('_')[0].replace('bitQuant', '')) if 'bitQuant' in x else 16) 
    # TODO: currently cannot support MixQuant, find a proper way
    df['method_name'] = df['method_name'].apply(lambda x: '_'.join(x.split('_')[1:]) if 'bitQuant' in x else x)

    # build summary table of chosen items
    # calculate the summation of accuracy_loss and latency_cost of each matrix name and method name
    used_columns = ['matrix_name', 'method_name', 'accuracy_loss', 'latency_cost', 'chosen', 'primary_config', 'bitwidth']

    df_summary = df[used_columns][df['chosen'] == 1].groupby(['matrix_name', 'method_name']).sum()[['accuracy_loss', 'latency_cost', 'chosen']]
    df_summary.columns = [col + '_sum' for col in df_summary.columns]

    df_mean = df[used_columns][df['chosen'] == 1].groupby(['matrix_name', 'method_name']).mean()[['accuracy_loss', 'latency_cost', 'primary_config', 'bitwidth']]
    df_mean.columns = [col + '_mean' for col in df_mean.columns]

    df_min = df[used_columns][df['chosen'] == 1].groupby(['matrix_name', 'method_name']).min()[['accuracy_loss', 'latency_cost', 'primary_config', 'bitwidth']]
    df_min.columns = [col + '_min' for col in df_min.columns]

    df_max = df[used_columns][df['chosen'] == 1].groupby(['matrix_name', 'method_name']).max()[['accuracy_loss', 'latency_cost', 'primary_config', 'bitwidth']]
    df_max.columns = [col + '_max' for col in df_max.columns]
    
    print("analysis")
    df_mean_primary = df[['chosen', 'matrix_name', 'primary_config']][df['chosen'] == 1].groupby(['matrix_name']).mean()[['primary_config']]
    df_mean_primary.rename(columns={'primary_config': 'primary_config_mean'}, inplace=True)
    print(df_mean_primary.round(2))
    print()

    # concat df_summary and df_mean, add a row to show the summation of each column
    df_analysis = pd.concat([df_summary, df_mean], axis=1)
    # Calculate the sum of each column and create a DataFrame with a multi-level index
    sum_df = pd.DataFrame(df_analysis.sum()).T
    sum_df.index = pd.MultiIndex.from_tuples([('Total', '')])
    # Concatenate the sum_df with df_analysis
    df_analysis = pd.concat([df_analysis, sum_df])
    # Print the DataFrame, rounding numerical values to two decimal places
    print(df_analysis.round(2))
    print()

    # reset index
    df_analysis = pd.concat([df_summary, df_mean, df_min, df_max], axis=1)
    df_analysis = df_analysis.reset_index()

    return df, df_analysis

def combine_summary_table(weight_prune_summary_table, quant_summary_table):
    """
    Combine the summary table of weight prune and quantization.
    """
    combined_table = deepcopy(weight_prune_summary_table)
    combined_table['bitwidth'] = quant_summary_table['bitwidth']
    combined_table['quant_accuracy_loss'] = quant_summary_table['accuracy_loss']
    combined_table['origin_accuracy_loss'] = weight_prune_summary_table['accuracy_loss']
    combined_table['origin_latency'] = weight_prune_summary_table['latency']
    return combined_table

def config_dataframe_to_json(dataframe: pd.DataFrame):
    """

    Example: 
        df = pd.read_csv(csv_file_path)
        json = config_dataframe_to_json(df)
        with open(output_json_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

    """

    # Extract unique layer identifiers
    layers = dataframe['layer_id'].unique()

    # Create the JSON structure
    json_data = {
        'alphas': [],
        'betas': []
    }

    # Populate the JSON data structure
    for layer in layers:
        layer_data = dataframe[dataframe['layer_id'] == layer]
        heads = layer_data['head_id'].unique()
        
        layer_alphas = []
        layer_betas = []
        
        for head in heads:
            head_data = layer_data[layer_data['head_id'] == head]
            layer_alphas.append(head_data['alpha_value'].values[0])
            layer_betas.append(head_data['beta_value'].values[0])
        
        json_data['alphas'].append(layer_alphas)
        json_data['betas'].append(layer_betas)

    return json_data
