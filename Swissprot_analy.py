import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os

def parse_active_site_residues(text):
    """
    Parses a string containing active site information and extracts 
    the residue numbers following the 'ACT_SITE' tag.
    
    Args:
        text (str): The string from the "Active site" column in UniProt data.
        
    Returns:
        list: A list of integers representing the residue numbers found.
    """
    if not isinstance(text, str) or pd.isna(text):
        return []
    
    # Matches 'ACT_SITE' followed by whitespace and the residue position
    pattern = r'ACT_SITE\s+(\d+)'
    matches = re.findall(pattern, text)
    
    list_of_activesites = [int(m) for m in matches]
    unique_activesites = list(set(list_of_activesites))
    
    return unique_activesites

def filter_uniprot_data(df, min_len=100, max_len=1000):
    """
    Filters the DataFrame based on EC numbers, Rhea IDs, and sequence length.

    Args:
        df (pd.DataFrame): The raw UniProt DataFrame.
        min_len (int): Minimum protein sequence length.
        max_len (int): Maximum protein sequence length.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only rows with active sites.
    """
    # Filtering for valid enzymatic data and sequence length constraints
    mask = (
        df["EC number"].notna() & 
        df["Rhea ID"].notna() & 
        (df["Length"] > min_len) & 
        (df["Length"] < max_len)
    )
    df_filtered = df[mask].copy()
    
    # Extract active site positions and filter for entries that actually have them
    df_filtered['ACT_SITE_list'] = df_filtered['Active site'].apply(parse_active_site_residues)
    df_filtered_act = df_filtered[df_filtered['ACT_SITE_list'].map(len) > 0]
    
    return df_filtered.reset_index(drop=True), df_filtered_act.reset_index(drop=True)

def plot_active_site_distribution(df, output_path='active_sites_distribution.png'):
    """
    Generates and saves a bar chart showing the distribution of active site counts.

    Args:
        df (pd.DataFrame): DataFrame containing the 'ACT_SITE_list' column.
        output_path (str): File path to save the generated plot.
    """
    # Calculate counts per protein
    df['site_count'] = df['ACT_SITE_list'].map(len)
    counts_distribution = df['site_count'].value_counts().sort_index()

    # Visualization setup
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts_distribution.index, y=counts_distribution.values, palette='viridis', edgecolor='black')

    plt.title('Distribution of Active Site Counts per Protein', fontsize=14)
    plt.xlabel('Number of Active Sites', fontsize=12)
    plt.ylabel('Frequency (Number of Proteins)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

def plot_ec_distribution_by_active_site(df_filtered, df_filtered_act, output_path='ec_distribution_by_active_site.png'):
    """
    Plots the EC number distribution comparing proteins with and without
    active site annotations.

    Args:
        df_filtered (pd.DataFrame): All filtered proteins (with and without active sites).
        df_filtered_act (pd.DataFrame): Only proteins WITH active site annotations.
        output_path (str): File path to save the generated plot.
    """
    def extract_ec_class(ec_str):
        """Extracts the top-level EC class (e.g., '1' from '1.2.3.4')."""
        if not isinstance(ec_str, str):
            return None
        # Takes the first EC number if multiple exist, then gets the first digit (class)
        first_ec = ec_str.split(';')[0].strip()
        match = re.match(r'^(\d+)', first_ec)
        return match.group(1) if match else None

    ec_labels = {
        '1': 'EC 1\nOxidoreductases',
        '2': 'EC 2\nTransferases',
        '3': 'EC 3\nHydrolases',
        '4': 'EC 4\nLyases',
        '5': 'EC 5\nIsomerases',
        '6': 'EC 6\nLigases',
        '7': 'EC 7\nTranslocases'
    }

    # Identify proteins WITHOUT active site annotation
    ids_with_site = set(df_filtered_act.index)
    df_no_site = df_filtered[~df_filtered.index.isin(ids_with_site)].copy()

    # Extract EC class for both groups
    df_filtered_act = df_filtered_act.copy()
    df_filtered_act['EC_class'] = df_filtered_act['EC number'].apply(extract_ec_class)
    df_no_site['EC_class'] = df_no_site['EC number'].apply(extract_ec_class)

    counts_with = df_filtered_act['EC_class'].value_counts().sort_index()
    counts_without = df_no_site['EC_class'].value_counts().sort_index()

    # Align indices
    all_classes = sorted(set(counts_with.index) | set(counts_without.index))
    counts_with = counts_with.reindex(all_classes, fill_value=0)
    counts_without = counts_without.reindex(all_classes, fill_value=0)

    x = range(len(all_classes))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar([i - width/2 for i in x], counts_with.values, width, label='With Active Site', color='steelblue', edgecolor='black')
    bars2 = ax.bar([i + width/2 for i in x], counts_without.values, width, label='Without Active Site', color='salmon', edgecolor='black')

    ax.set_xticks(list(x))
    ax.set_xticklabels([ec_labels.get(c, f'EC {c}') for c in all_classes], fontsize=10)
    ax.set_title('EC Class Distribution: With vs Without Active Site Annotation', fontsize=14)
    ax.set_xlabel('EC Class', fontsize=12)
    ax.set_ylabel('Number of Proteins', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Add value labels on top of bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(int(bar.get_height())), ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(int(bar.get_height())), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"EC distribution plot saved to: {output_path}")


def summarize_ec_counts_with_active_site(df_filtered_act, output_path='ec_counts_active_site.csv'):
    """
    Counts unique full EC numbers among proteins WITH active site annotations,
    saves a CSV summary, and prints the top results.

    Args:
        df_filtered_act (pd.DataFrame): Proteins with active site annotations.
        output_path (str): Path to save the CSV summary.

    Returns:
        pd.DataFrame: Summary table with EC number counts.
    """
    # Explode multiple EC numbers (some entries have several separated by '; ')
    ec_series = df_filtered_act['EC number'].dropna().str.split(';').explode().str.strip()
    ec_counts = ec_series.value_counts().reset_index()
    ec_counts.columns = ['EC_number', 'count']
    ec_counts = ec_counts.sort_values('count', ascending=False).reset_index(drop=True)

    ec_counts.to_csv(output_path, index=False)
    print(f"\nEC counts summary saved to: {output_path}")
    print(f"Total unique EC numbers (with active site): {len(ec_counts)}")
    print("\nTop 20 EC numbers:")
    print(ec_counts.head(20).to_string(index=False))

    return ec_counts



input_file = '/home/joao/Documents/Doutorado/Benchmarks/Databases/SwissProt/uniprotkb_AND_reviewed_true_2026_03_10.tsv'

if not os.path.exists(input_file):
    print(f"Error: File {input_file} not found.")


# Load data
print("Loading dataset...")
df = pd.read_csv(input_file, sep='\t')

# Process and Filter
print("Filtering proteins and parsing active sites...")
processed_df, processed_df_act = filter_uniprot_data(df)

# Visualize
print("Generating distribution plot...")
plot_active_site_distribution(processed_df)

print(f"Analysis complete. Total proteins processed: {len(processed_df)}")




processed_df_act.columns

higher_3_active = processed_df_act[processed_df_act["ACT_SITE_list"].apply(len) == 4]


# New Step 1: EC distribution — with vs without active site
print("Generating EC distribution by active site annotation...")
plot_ec_distribution_by_active_site(processed_df, processed_df_act)

# New Step 2: EC count summary for proteins WITH active site
print("Summarizing EC numbers for proteins with active site...")
ec_summary = summarize_ec_counts_with_active_site(processed_df_act)



####################################################

subset_with_catalytics = processed_df[processed_df["site_count"] > 0]
subset_with_catalytics['ACT_SITE_list'] = subset_with_catalytics['ACT_SITE_list'].apply(lambda x: ';'.join(map(str, x)))
subset_with_catalytics.to_csv('catalytic_sites.csv', index=False)


