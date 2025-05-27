import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
import re # For sanitizing filenames
import numpy as np # For handling potential NaN in x-axis for discrete scores

# Constants for source names
SOURCE_NAME_INPUT = "openai"
SOURCE_NAME_GENERATED = "bud-qwen3-4b" # Corrected typo from your last request if intended

def load_and_combine_json_data(file_paths):
    """
    Loads data from multiple JSON files and combines them.
    """
    combined_data = {}
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for query, outfit_ratings in data.items():
                if query in combined_data:
                    combined_data[query].extend(outfit_ratings)
                else:
                    combined_data[query] = outfit_ratings
        except FileNotFoundError:
            print(f"Warning: File not found {file_path}")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}")
        except Exception as e:
            print(f"Warning: An error occurred while processing {file_path}: {e}")
    return combined_data

def sanitize_filename(filename):
    """Removes or replaces characters that are problematic in filenames."""
    filename = re.sub(r'[^\w\s-]', '', filename).strip()
    filename = re.sub(r'[-\s]+', '_', filename)
    return filename[:100]


def extract_scores_to_dataframe(data):
    """
    Extracts scores from the combined JSON data into a Pandas DataFrame.
    """
    records = []
    score_types = ["closeness", "relevance", "helpfulness", "quality"]

    for query, outfit_pairs in data.items():
        if not isinstance(outfit_pairs, list): 
            print(f"Warning: Skipping query '{query}' because its value is not a list (type: {type(outfit_pairs)}).")
            continue
        if not outfit_pairs:
            continue

        for pair_idx, pair_data in enumerate(outfit_pairs):
            if not isinstance(pair_data, dict):
                print(f"Warning: Skipping an item in query '{query}' because it's not a dictionary (type: {type(pair_data)}, value: {pair_data}).")
                continue

            record = {
                "query": query,
                "pair_index": pair_idx,
                "input_collection": pair_data.get("input_collection", "N/A"),
                "generated_outfit_id": pair_data.get("generated_outfit_id", "N/A"),
                "similarity": pair_data.get("similarity", None)
            }
            for score_type in score_types:
                input_col_prefix = f"{SOURCE_NAME_INPUT}_{score_type}" 
                generated_col_prefix = f"{SOURCE_NAME_GENERATED}_{score_type}"

                input_score_key = f"{score_type}_input" 
                input_score_data = pair_data.get(input_score_key, {})
                if isinstance(input_score_data, dict):
                    record[input_col_prefix] = input_score_data.get(score_type, None)
                    record[f"{input_col_prefix}_reason"] = input_score_data.get("reason", "N/A")
                else:
                    record[input_col_prefix] = None
                    record[f"{input_col_prefix}_reason"] = "N/A (invalid data format)"


                generated_score_key = f"{score_type}_generated" 
                generated_score_data = pair_data.get(generated_score_key, {})
                if isinstance(generated_score_data, dict):
                    record[generated_col_prefix] = generated_score_data.get(score_type, None)
                    record[f"{generated_col_prefix}_reason"] = generated_score_data.get("reason", "N/A")
                else:
                    record[generated_col_prefix] = None
                    record[f"{generated_col_prefix}_reason"] = "N/A (invalid data format)"
            
            records.append(record)
            
    df = pd.DataFrame(records)
    
    df["similarity"] = pd.to_numeric(df["similarity"], errors='coerce')
    for score_type in score_types:
        df[f"{SOURCE_NAME_INPUT}_{score_type}"] = pd.to_numeric(df[f"{SOURCE_NAME_INPUT}_{score_type}"], errors='coerce')
        df[f"{SOURCE_NAME_GENERATED}_{score_type}"] = pd.to_numeric(df[f"{SOURCE_NAME_GENERATED}_{score_type}"], errors='coerce')
        
    return df


def plot_overall_average_scores(df, output_directory="output_graphs"):
    """
    Generates a single bar plot for the overall average scores across all queries
    and saves it.
    """
    if df.empty:
        print("DataFrame is empty. No overall average plot will be generated.")
        return

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")

    score_types = ["closeness", "relevance", "helpfulness", "quality"]
    overall_avg_scores_data = []
    total_outfit_pairs = len(df)

    for score_type in score_types:
        avg_input = df[f"{SOURCE_NAME_INPUT}_{score_type}"].mean()
        avg_generated = df[f"{SOURCE_NAME_GENERATED}_{score_type}"].mean()
        
        if pd.notna(avg_input):
             overall_avg_scores_data.append({
                "Metric": score_type.capitalize(),
                "Source": SOURCE_NAME_INPUT,
                "Average Score": avg_input
            })
        if pd.notna(avg_generated):
            overall_avg_scores_data.append({
                "Metric": score_type.capitalize(),
                "Source": SOURCE_NAME_GENERATED,
                "Average Score": avg_generated
            })
    
    if not overall_avg_scores_data:
        print("No valid scores to plot for overall average.")
        return

    plot_df = pd.DataFrame(overall_avg_scores_data)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Metric", y="Average Score", hue="Source", data=plot_df, 
                     palette={SOURCE_NAME_INPUT: "skyblue", SOURCE_NAME_GENERATED: "lightcoral"}) 
    
    for p in ax.patches:
        if p.get_height() > 0:
             ax.annotate(f"{p.get_height():.3f}",
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='center', 
                         xytext=(0, 5), 
                         textcoords='offset points',
                         fontsize=9)

    title_text = f"Overall Average Scores: {SOURCE_NAME_INPUT} vs. {SOURCE_NAME_GENERATED}"
    plt.title(f"{title_text}\n(Based on {total_outfit_pairs} outfit pairs from {df['query'].nunique()} queries)", fontsize=14)
    plt.ylabel("Average Score (0-1)", fontsize=12)
    plt.xlabel("Metric", fontsize=12)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Outfit Source")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_filename = os.path.join(output_directory, "overall_average_scores.png")
    try:
        plt.savefig(output_filename)
        print(f"Saved plot: {output_filename}")
    except Exception as e:
        print(f"Error saving plot {output_filename}: {e}")
    
    plt.close()

def plot_similarity_distribution(df, output_directory="output_graphs"):
    """Plots the distribution of similarity scores."""
    if df.empty or 'similarity' not in df.columns or df['similarity'].isna().all():
        print("No valid similarity data to plot.")
        return

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")

    similarity_counts = df['similarity'].value_counts().sort_index()
    if similarity_counts.empty:
        print("Similarity counts are empty.")
        return

    plt.figure(figsize=(10, 6))
    ax = similarity_counts.plot(kind='bar', color='mediumpurple')
    plt.title(f"Distribution of Similarity Scores\n({SOURCE_NAME_INPUT} vs. {SOURCE_NAME_GENERATED} Outfits, N={df['similarity'].count()})", fontsize=14)
    plt.xlabel("Similarity Score", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for p in ax.patches:
        ax.annotate(str(int(p.get_height())), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points',
                    fontsize=9)
    
    plt.tight_layout()
    output_filename = os.path.join(output_directory, "similarity_score_distribution.png")
    try:
        plt.savefig(output_filename)
        print(f"Saved plot: {output_filename}")
    except Exception as e:
        print(f"Error saving plot {output_filename}: {e}")
    plt.close()


def plot_metric_score_distributions_comparison(df, output_directory="output_graphs"):
    """
    Plots the distribution of scores for each metric (closeness, relevance, etc.)
    comparing openai vs. bud-qwen3-4b using a grouped bar chart.
    """
    if df.empty:
        print("DataFrame is empty. No detailed metric distributions will be plotted.")
        return

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")

    score_types = ["closeness", "relevance", "helpfulness", "quality"]

    # Determine common possible scores across all metrics for consistent x-axis
    all_possible_scores_list = []
    for score_type in score_types:
        input_col = f"{SOURCE_NAME_INPUT}_{score_type}"
        generated_col = f"{SOURCE_NAME_GENERATED}_{score_type}"
        if input_col in df.columns:
            all_possible_scores_list.extend(df[input_col].dropna().unique())
        if generated_col in df.columns:
             all_possible_scores_list.extend(df[generated_col].dropna().unique())
    
    if not all_possible_scores_list:
        print("No valid numeric scores found for any metric distributions.")
        return
    # Ensure scores are floats and sorted for the x-axis order
    common_score_order = sorted(list(set(float(s) for s in all_possible_scores_list if pd.notna(s))))


    for score_type in score_types:
        input_col = f"{SOURCE_NAME_INPUT}_{score_type}"
        generated_col = f"{SOURCE_NAME_GENERATED}_{score_type}"

        # Check if columns exist and have data
        if not (input_col in df.columns and generated_col in df.columns and \
                (not df[input_col].isna().all() or not df[generated_col].isna().all())):
            print(f"Skipping distribution plot for '{score_type}' due to missing or all-NaN data.")
            continue
        
        # Prepare data for plotting: count occurrences of each score for input and generated
        input_counts = df[input_col].value_counts().reindex(common_score_order, fill_value=0)
        generated_counts = df[generated_col].value_counts().reindex(common_score_order, fill_value=0)

        plot_data = []
        for score_val in common_score_order:
            plot_data.append({"Score": score_val, "Source": SOURCE_NAME_INPUT, "Count": input_counts.get(score_val, 0)})
            plot_data.append({"Score": score_val, "Source": SOURCE_NAME_GENERATED, "Count": generated_counts.get(score_val, 0)})
        
        plot_df_metric = pd.DataFrame(plot_data)
        
        if plot_df_metric.empty:
            print(f"No data to plot for metric distribution comparison: {score_type}")
            continue

        plt.figure(figsize=(12, 7))
        ax = sns.barplot(x='Score', y='Count', hue='Source', data=plot_df_metric,
                           palette={SOURCE_NAME_INPUT: "skyblue", SOURCE_NAME_GENERATED: "lightcoral"})

        plt.title(f"Distribution of {score_type.capitalize()} Scores\n({SOURCE_NAME_INPUT} vs. {SOURCE_NAME_GENERATED})", fontsize=14)
        plt.xlabel(f"{score_type.capitalize()} Score Value", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.legend(title="Outfit Source")
        
        # Format x-axis ticks to show one decimal place
        ax.set_xticklabels([f"{score:.1f}" for score in common_score_order], rotation=45, ha="right")
        
        for p in ax.patches:
            if p.get_height() > 0 : 
                 ax.annotate(f'{int(p.get_height())}', 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha = 'center', va = 'center', 
                             xytext = (0, 5), 
                             textcoords = 'offset points',
                             fontsize=9)

        plt.tight_layout()
        output_filename = os.path.join(output_directory, f"{sanitize_filename(score_type)}_comparison_dist.png")
        try:
            plt.savefig(output_filename)
            print(f"Saved plot: {output_filename}")
        except Exception as e:
            print(f"Error saving plot {output_filename}: {e}")
        plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    current_directory = os.getcwd()
    # json_file_paths = glob(os.path.join(current_directory, "*.json")) 
    json_file_paths = ["rated_outfits_results.json"]
    output_dir = "output_score_analysis_v4_comparison"

    if not json_file_paths:
        print("No JSON files found in the specified location.")
    else:
        print(f"Processing JSON files: {json_file_paths}")
        
        combined_data = load_and_combine_json_data(json_file_paths)
        
        if not combined_data:
            print("No data loaded from JSON files.")
        else:
            scores_df = extract_scores_to_dataframe(combined_data)
            
            if scores_df.empty:
                print("DataFrame is empty after extraction. Exiting.")
            else:
                plot_overall_average_scores(scores_df, output_directory=output_dir)
                plot_similarity_distribution(scores_df, output_directory=output_dir)
                # Call the corrected function for metric distribution comparison
                plot_metric_score_distributions_comparison(scores_df, output_directory=output_dir)
                
                print(f"\nAll visualizations saved to '{output_dir}' directory.")