import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(data):
    # Distribution of features
    ig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))

    sns.histplot(data['popularity'], kde=True, ax=axes[0], bins=30, color='skyblue')
    axes[0].set_title('Distribution of Popularity')
    axes[0].set_xlabel('Popularity')
    axes[0].set_ylabel('Frequency')

    sns.histplot(data['danceability'], kde=True, ax=axes[1], bins=30, color='salmon')
    axes[1].set_title('Distribution of Danceability')
    axes[1].set_xlabel('Danceability')
    axes[1].set_ylabel('Frequency')

    sns.histplot(data['energy'], kde=True, ax=axes[2], bins=30, color='limegreen')
    axes[2].set_title('Distribution of Energy')
    axes[2].set_xlabel('Energy')
    axes[2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('outputs/figures/feature_distributions.png')
    
    # Boxplots for popularity against key and mode
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))

    sns.boxplot(x='key', y='popularity', data=data, ax=axes[0], palette="viridis")
    axes[0].set_title('Popularity across Different Keys')
    axes[0].set_xlabel('Key')
    axes[0].set_ylabel('Popularity')

    sns.boxplot(x='mode', y='popularity', data=data, ax=axes[1], palette="viridis")
    axes[1].set_title('Popularity across Modes (Major/Minor)')
    axes[1].set_xlabel('Mode (0: Minor, 1: Major)')
    axes[1].set_ylabel('Popularity')

    plt.tight_layout()
    plt.savefig('outputs/figures/popularity_key_mode.png')

    # Filter out columns with the prefix "Unnamed"
    filtered_columns = [col for col in data.columns if not col.startswith('Unnamed')]
    filtered_dataset = data[filtered_columns]

    # Calculate the correlation matrix
    correlation_matrix = filtered_dataset.corr()

    # Plotting the correlation heatmap
    plt.figure(figsize=(16, 12), dpi=300)
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, linecolor="black", vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.savefig('outputs/figures/correlation_heatmap.png')
    
    
    
    # Create a subplot of 3 plots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))

    # 1. Popularity Distribution (KDE plot)
    sns.kdeplot(data['popularity'], fill=True, ax=axes[0, 0], color='purple')
    axes[0, 0].set_title('Distribution of Song Popularity')
    axes[0, 0].set_xlabel('Popularity')
    axes[0, 0].set_ylabel('Density')

    # 2. Energy vs. Danceability Scatterplot
    sns.scatterplot(x='danceability', y='energy', hue='popularity', size='popularity', sizes=(10, 200), data=data, ax=axes[0, 1], palette="viridis", alpha=0.6)
    axes[0, 1].set_title('Energy vs. Danceability (Colored by Popularity)')
    axes[0, 1].set_xlabel('Danceability')
    axes[0, 1].set_ylabel('Energy')

    # 3. Loudness vs. Acousticness Hexbin plot
    hb = axes[1, 0].hexbin(data['loudness'], data['acousticness'], gridsize=50, cmap='Blues')
    cb = plt.colorbar(hb, ax=axes[1, 0])
    cb.set_label('Counts')
    axes[1, 0].set_title('Loudness vs. Acousticness')
    axes[1, 0].set_xlabel('Loudness')
    axes[1, 0].set_ylabel('Acousticness')

    # Remove the unused subplot
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('outputs/figures/eda_plots.png')
