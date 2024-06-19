import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot_correct_rate_heatmap_input_length_position(df, output_path):
    """
    Generates and displays a heatmap showing the correct rate by input length and relative position percentage.
    
    Parameters:
    - df: DataFrame containing the columns 'is_correct', 'num_lines', 'key_id', and 'length_level'.
    """
    # Convert 'is_correct' from bool to int for easier aggregation
    df['is_correct'] = df['is_correct'].astype(int)

    # Calculate the relative position and bin it into 10 bins
    df['relative_position'] = df['key_id'].astype(float) / df['num_lines'].astype(float)
    bin_edges = [i * 0.1 for i in range(11)] 
    df['relative_position_bin'] = pd.cut(df['relative_position'], bins=bin_edges, labels=False, right=False)

    # Calculate the correct rate for each combination of input length and relative position bin
    correct_rate_binned_df = df.groupby(['length_level', 'relative_position_bin'])['is_correct'].mean().reset_index()

    # Pivot the data for the heatmap
    pivot_table_binned = correct_rate_binned_df.pivot(index="relative_position_bin", columns="length_level", values="is_correct")

    # Plot the heatmap
    plt.figure(figsize=(14, 8))
    ax = sns.heatmap(pivot_table_binned, cmap="RdYlGn", annot=True, fmt=".1f", vmin=0.0, vmax=1.0, annot_kws={"fontsize":24})
    plt.title("Correct Rate by Input Length and Relative Position (%)")
    plt.xlabel("Input Length")
    plt.ylabel("Relative Position (%)")

    # Create percentage labels for the y-axis based on the binning
    bin_labels = [f"{int(i*10)}-{int((i+1)*10)}%" for i in range(10)]
    plt.yticks(np.arange(10) + 0.5, labels=bin_labels, rotation=0)

    # Configure the colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label('Correct Rate', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(output_path)

def plot_correct_rate_heatmap_input_length(df, output_path):
    # Calculate the correct rate and count by length_level
    correct_rate_data = df.groupby('length_level')['is_correct'].mean().reset_index()
    data_point_count = df.groupby('length_level').size().reset_index(name='count')

    # Sort by length_level to ensure the heatmap displays in order
    correct_rate_data.sort_values('length_level', inplace=True)

    # Create a horizontal heatmap
    plt.figure(figsize=(12, 0.8))  # Adjusting the figure size for a 1*x heatmap
    sns.heatmap(correct_rate_data[['is_correct']].T, cmap="RdYlGn", annot=True, cbar=False,
                xticklabels=correct_rate_data['length_level'].values, vmin=0.0, vmax=1.0)
    plt.title('Correct Rate by Input Length')
    plt.yticks([])  # Hide y-ticks as we have only one row
    plt.xlabel('Input Length')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    # Return the DataFrames
    return correct_rate_data, data_point_count

def plot_data_count_heatmap_input_length_position(df, output_path):
    """
    Generates and displays a heatmap showing the distribution of data points by input length and relative position percentage.
    
    Parameters:
    - df: DataFrame containing the columns 'num_lines', 'key_id', and 'length_level'.
    """
    # Calculate the relative position and bin it into 10 bins
    df['relative_position'] = df['key_id'].astype(float) / df['num_lines'].astype(float)
    bin_edges = [i * 0.1 for i in range(11)] 
    df['relative_position_bin'] = pd.cut(df['relative_position'], bins=bin_edges, labels=False, right=False)

    # Count the number of data points in each combination of input length and binned relative position
    data_point_count_df = df.groupby(['length_level', 'relative_position_bin']).size().reset_index(name='count')

    # Pivot the data for the heatmap
    pivot_table_count = data_point_count_df.pivot(index="relative_position_bin", columns="length_level", values="count")

    # Plot the heatmap for the count of data points
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(pivot_table_count, cmap="BuPu", annot=True, fmt=".0f")
    plt.title("Number of Data Points by Input Length and Relative Position (%)")
    plt.xlabel("Input Length")
    plt.ylabel("Relative Position (%)")

    # Create percentage labels for the y-axis based on the binning
    bin_labels = [f"{int(i*10)}-{int((i+1)*10)}%" for i in range(10)]
    plt.yticks(np.arange(10) + 0.5, labels=bin_labels, rotation=0)

    # Configure the colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label('Number of Data Points', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_correct_rate_heatmap_input_length_attention_span(df, output_path):
    """
    Generates and saves a heatmap showing the retrieval accuracy by input length and attention span.
    
    Parameters:
    - df: DataFrame containing the columns 'is_correct', 'length_level', and 'context_length'.
    """
    
    # Convert 'is_correct' from bool to int for easier aggregation
    df['is_correct'] = df['is_correct'].astype(int)
    
    # Group by input length and attention span, then calculate the mean correct rate
    correct_rate_df = df.groupby(['length_level', 'context_length'])['is_correct'].mean().reset_index()

    # Pivot the data for the heatmap, sorting the columns in ascending order
    pivot_table = correct_rate_df.pivot(index="length_level", columns="context_length", values="is_correct")

    # For the pivot table, if context length > length level, fill the is correct ratio value with the value of context length = length level
    for i in range(len(pivot_table)):
        for j in range(i, len(pivot_table.columns)):
            pivot_table.iloc[i, j] = pivot_table.iloc[i, i]

    breakpoint()
    
    # Reverse the order of the rows in the pivot table to start the y-axis from the lower-left corner
    pivot_table = pivot_table.iloc[::-1]

    # Create a mask where attention span is greater than input length
    mask = pivot_table.index.values[:, None] >= pivot_table.columns.values
    # Fill masked areas with -1 before plotting
    pivot_table = pivot_table.fillna(-1)

    # # Update mask to cover cells filled with -1 for grey coloring
    # mask = (mask) | (pivot_table == -1)

    # Plot the heatmap
    # mask = None # noqa
    plt.figure(figsize=(12, 8))
    if mask is not None:
        ax = sns.heatmap(pivot_table, cmap="RdYlGn", annot=True, fmt=".2f", mask=~mask, vmin=0.0, vmax=1.0, cbar_kws={'label': 'Retrieval Accuracy'})
        sns.heatmap(pivot_table, cmap=["#D3D3D3"], annot=False, mask=mask, cbar=False, ax=ax)  # No annotation for the grey cells
    else:
        ax = sns.heatmap(pivot_table, cmap="RdYlGn", annot=True, fmt=".2f", vmin=0.0, vmax=1.0, cbar_kws={'label': 'Retrieval Accuracy'})

    plt.title("Retrieval Accuracy")
    plt.xlabel("Average Attention Span")
    plt.ylabel("Context Length")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    # Count the number of data points in each combination of input length and attention span
    data_point_count_df = df.groupby(['length_level', 'context_length']).size().reset_index(name='count')

    return correct_rate_df, data_point_count_df


def plot_correct_rate_heatmap_input_length_attention_span_from_ready_data(df, output_path):
    """
    Generates and saves a heatmap showing the retrieval accuracy by input length and attention span.
    
    Parameters:
    - df: DataFrame containing the columns 'is_correct', 'length_level', and 'context_length'.
    """
    
    pivot_table = df
    pivot_table.set_index('length_level', inplace=True)
    # Add an extra level to the column headers

    
    # Reverse the order of the rows in the pivot table to start the y-axis from the lower-left corner
    pivot_table = pivot_table.iloc[::-1]

    # Create a mask where attention span is greater than input length
    columns = pivot_table.columns
    # delete "length_level" from the list
    columns = columns[columns != 'length_level']
    # turn columns to int
    columns = columns.astype(int)
    mask = pivot_table.index.values[:, None] >= columns.values
    # Fill masked areas with -1 before plotting
    pivot_table = pivot_table.fillna(-1)

    # # Update mask to cover cells filled with -1 for grey coloring
    # mask = (mask) | (pivot_table == -1)

    # Plot the heatmap
    # mask = None # noqa
    plt.figure(figsize=(12, 8))
    if mask is not None:
        ax = sns.heatmap(pivot_table, cmap="RdYlGn", annot=True, fmt=".2f", mask=~mask, vmin=0.0, vmax=1.0, cbar_kws={'label': 'Retrieval Accuracy'}, annot_kws={"fontsize":18})
        sns.heatmap(pivot_table, cmap=["#D3D3D3"], annot=False, mask=mask, cbar=False, ax=ax)  # No annotation for the grey cells
    else:
        ax = sns.heatmap(pivot_table, cmap="RdYlGn", annot=True, fmt=".2f", vmin=0.0, vmax=1.0, cbar_kws={'label': 'Retrieval Accuracy'}, annot_kws={"fontsize":18})

    plt.title("(b) MoA", fontsize=22)
    plt.xlabel("Attention Span", fontsize=18)
    plt.ylabel("Input Length", fontsize=18)
    # Set larger tick parameters
    ax.tick_params(axis='both', which='major', labelsize=18)  # 'major' applies to major ticks

    # Access and modify the color bar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)  # Set font size of the color bar's tick labels
    cbar.set_label('Retrieval Accuracy', fontsize=18)  # Set font size of the color bar's label

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_correct_rate_facet_grid(df, output_path):
    """
    Generates and saves a facet grid of heatmaps showing the correct rate by input length, relative position,
    and attention span, with a shared colorbar for all heatmaps.

    Parameters:
    - df: DataFrame containing the necessary data
    - output_path: Path where the generated figure will be saved
    """
    # Prepare the data
    df['relative_position'] = df['key_id'].astype(float) / df['num_lines'].astype(float)
    bin_edges = [i * 0.1 for i in range(11)]
    df['relative_position_bin'] = pd.cut(df['relative_position'], bins=bin_edges, labels=False, right=False)
    df['is_correct'] = df['is_correct'].astype(int)
    correct_rate_df = df.groupby(['length_level', 'relative_position_bin', 'context_length'])['is_correct'].mean().reset_index()
    
    # Determine the layout
    rows, cols = 2, 4 # noqa
    unique_context_lengths = correct_rate_df['context_length'].unique()
    fig, axs = plt.subplots(rows, cols, figsize=(24, 12), constrained_layout=True)
    axs_flat = axs.flatten()

    # Generate heatmaps
    for i, context_length in enumerate(sorted(unique_context_lengths)):
        if i < len(axs_flat):
            sns.heatmap(correct_rate_df[correct_rate_df['context_length'] == context_length].pivot(index="relative_position_bin", columns="length_level", values="is_correct"), 
                        cmap="RdYlGn", annot=True, fmt=".1f", ax=axs_flat[i], vmin=0, vmax=1, cbar=False)
            axs_flat[i].set_title(f"Attention Span: {context_length}")
            axs_flat[i].set_xlabel("Input Length")
            axs_flat[i].set_ylabel("Relative Position Bin (%)")
            bin_labels = [f"{int(i*10)}-{int((i+1)*10)}%" for i in range(10)]
            axs_flat[i].set_yticklabels(bin_labels, rotation=0)
        else:
            break

    # Hide any unused subplots
    for ax in axs_flat[i+1:]:
        ax.set_visible(False)

    # Add a single, shared colorbar
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=axs, location='right', fraction=0.01, pad=0.02)

    plt.suptitle('Correct Rate by Input Length, Relative Position, and Attention Span', fontsize=20)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()  # Close the plot to free up memory

def plot_effective_context_length(df):
    """
    Plots effective context length against attention span from a DataFrame that includes multiple named datasets.
    
    Parameters:
    - df: DataFrame containing columns 'context_length', 'length_level', 'is_correct', and 'name'.
    
    Each 'name' represents a different dataset to be plotted separately. Points are colored based on the 'is_correct'
    rate using the 'RdYlGn' colormap.
    """
    sns.set_theme(style="whitegrid")

    # Filter data to include only points where is_correct >= 0.9
    threashold = 0.9
    valid_data = df[df['is_correct'] >= threashold]

    # For each context_length and name, select the entry with the highest length_level and its corresponding is_correct
    optimal_data = valid_data.loc[valid_data.groupby(['context_length', 'name'])['length_level'].idxmax()]

    # To align color map, add (0,0) point with is_correct=0.0 and length_level=0
    # optimal_data = optimal_data.append({'context_length': 0.0, 'length_level': 0.0, 'is_correct': 0.0, 'name': df['name'].iloc[0]}, ignore_index=True)
    # optimal_data = optimal_data.append({'context_length': 0.0, 'length_level': 0.0, 'is_correct': 1.0, 'name': df['name'].iloc[0]}, ignore_index=True)


    # Prepare the plot
    plt.figure(figsize=(12, 8))
    # Define a set of markers and a color palette
    markers = ['o', 's', '^', 'P', 'D', 'X', '*']  # Extend with more markers as needed
    line_styles = ['--', '-', '-.', ':']  # Extend with more line styles as needed
    palette = sns.color_palette("RdYlGn", as_cmap=True)

    names = optimal_data['name'].unique()

    # Plot using Seaborn scatterplot and lineplot for continuity
    # rename is correct to Retrieve Accuracy 
    optimal_data = optimal_data.rename(columns={'is_correct': 'Retrieve Accuracy'})
    g = sns.scatterplot(data=optimal_data, x='context_length', y='length_level',hue='Retrieve Accuracy', style='name', markers=markers[:len(optimal_data['name'].unique())],size="Retrieve Accuracy", legend='full', palette=palette, hue_order=names, sizes=(50, 100))

    # remove the added (0,0) point
    # optimal_data = optimal_data[optimal_data['context_length'] != 0.0]

    

    # Map each name to a line style
    name_to_linestyle = dict(zip(names, line_styles[:len(names)]))

    # Group the data by 'name' and plot lines to connect the dots
    for name, group in optimal_data.groupby('name'):
        # Sort the group by 'context_length' to ensure lines connect in order
        group = group.sort_values(by='context_length')
        plt.plot(group['context_length'], group['length_level'], marker='', linestyle=name_to_linestyle[name], color='black')


    # Adding color bar for correctness rate
    norm = Normalize(vmin=threashold, vmax=1.0)
    sm = ScalarMappable(cmap=palette, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Retrieve Accuracy')

    # Adding labels, title
    plt.xlabel('Attention Span')
    plt.ylabel('Effective Context Length')
    plt.title('Effective Context Length and Attention Span of Different Methods')

    # show x and y label every 1024 interval
    plt.xticks(np.arange(0, 8193, 1024))
    plt.yticks(np.arange(0, 8193, 1024))

    # the x and y limit
    plt.xlim(512, 8192 + 512)
    plt.ylim(512, 8192 + 512)

    plt.savefig('effective_context_length.png')
    plt.close()




# if __name__ == "__main__":
def main():
    output_surfix = "png"
    upper_bound = 16384
    input_dir = f"local/universal/paper_MoA/retrieval/vicuna-7b-v1.5/"
    output_dir = f"local/universal/paper_MoA/retrieval/vicuna-7b-v1.5/{upper_bound}"

    """
    compressed correct rate
    """
    file_name = "StreamingLLM"
    df = pd.read_csv(os.path.join(input_dir, f"{file_name}.csv"))
    # select the input length_level between 0 and bound
    df = df[(df['length_level'] >= 0) & (df['length_level'] <= upper_bound)]
    df = df[(df['context_length'] >= 0) & (df['context_length'] <= upper_bound)]

    # print the unique length level and context length of the df
    # print(compressed_df['length_level'].unique())
    # print(compressed_df['context_length'].unique())
    correct_rate_df, data_point_count_df = plot_correct_rate_heatmap_input_length_attention_span(df, os.path.join(output_dir, f"{file_name}_correct_rate_heatmap_input_length_attention_span.{output_surfix}"))
    # save the data point count heatmap and correct rate heatmap
    correct_rate_df["name"] = file_name
    
    correct_rate_df.to_csv(os.path.join(output_dir, f"{file_name}_correct_rate_heatmap_input_length_attention_span.csv"), index=False)
    data_point_count_df.to_csv(os.path.join(output_dir, f"{file_name}_data_point_count_heatmap_input_length_attention_span.csv"), index=False)

    heat_map = correct_rate_df.pivot("context_length", "length_level", "is_correct")
    # reverse the row sequence
    heat_map = heat_map.iloc[::-1]
    heat_map.to_csv(os.path.join(output_dir, f"{file_name}_heat_map_input_length_attention_span.csv"), index=True)

    print(data_point_count_df)
    print(correct_rate_df)

    """
    raw correct rate
    """
    file_name = "Raw"
    df = pd.read_csv(os.path.join(input_dir, f"{file_name}.csv"))
    # select the input length_level between 0 and bound
    df = df[(df['length_level'] >= 0) & (df['length_level'] <= upper_bound)]
    correct_rate_df, data_point_count_df = plot_correct_rate_heatmap_input_length(df, os.path.join(output_dir, f"{file_name}_correct_rate_heatmap_input_length.{output_surfix}"))
    # save the data point count heatmap and correct rate heatmap
    correct_rate_df.to_csv(os.path.join(output_dir, f"{file_name}_correct_rate_heatmap_input_length_attention_span.csv"), index=False)
    data_point_count_df.to_csv(os.path.join(output_dir, f"{file_name}_data_point_count_heatmap_input_length_attention_span.csv"), index=False)

    print(data_point_count_df)
    print(correct_rate_df)

    print("done")
