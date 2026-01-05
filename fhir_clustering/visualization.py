"""
Visualization utilities for cluster analysis.
Optional module for plotting and visual interpretation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix


def plot_cluster_sizes(cluster_labels: np.ndarray, 
                       figsize: Tuple[int, int] = (10, 6),
                       title: str = "Cluster Size Distribution") -> plt.Figure:
    """
    Plot bar chart of cluster sizes.
    
    Args:
        cluster_labels: Cluster assignments for each patient
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    unique, counts = np.unique(cluster_labels, return_counts=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Separate noise cluster if present
    mask = unique != -1
    regular_clusters = unique[mask]
    regular_counts = counts[mask]
    
    colors = ['#1f77b4'] * len(regular_clusters)
    labels = [f"Cluster {c}" for c in regular_clusters]
    
    if -1 in unique:
        noise_count = counts[unique == -1][0]
        labels.append("Noise")
        regular_counts = np.append(regular_counts, noise_count)
        colors.append('#d62728')
    
    bars = ax.bar(range(len(regular_counts)), regular_counts, color=colors)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Patients')
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


def plot_explained_variance(explained_variance_ratio: np.ndarray,
                           figsize: Tuple[int, int] = (12, 5),
                           title: str = "Explained Variance") -> plt.Figure:
    """
    Plot explained variance for dimensionality reduction.
    
    Args:
        explained_variance_ratio: Variance ratio for each component
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    n_components = len(explained_variance_ratio)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Individual variance
    ax1.bar(range(n_components), explained_variance_ratio, color='steelblue')
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title(f'{title} - Individual')
    ax1.grid(axis='y', alpha=0.3)
    
    # Cumulative variance
    ax2.plot(range(n_components), cumulative_variance, marker='o', 
             linewidth=2, color='steelblue')
    ax2.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90%')
    ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95%')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title(f'{title} - Cumulative')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_2d_clusters(X: np.ndarray, 
                     cluster_labels: np.ndarray,
                     method: str = 'pca',
                     figsize: Tuple[int, int] = (10, 8),
                     title: str = "Patient Clusters (2D Projection)") -> plt.Figure:
    """
    Plot 2D visualization of clusters.
    
    Args:
        X: Patient feature matrix (n_patients, n_features)
        cluster_labels: Cluster assignments
        method: Projection method ('pca' or 'tsne')
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Project to 2D if needed
    if X.shape[1] > 2:
        if method == 'tsne':
            projector = TSNE(n_components=2, random_state=42)
            X_2d = projector.fit_transform(X)
        else:  # pca
            from sklearn.decomposition import PCA
            projector = PCA(n_components=2, random_state=42)
            X_2d = projector.fit_transform(X)
    else:
        X_2d = X
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)
    
    # Create color palette
    n_clusters = len(unique_clusters[unique_clusters != -1])
    colors = sns.color_palette('husl', n_clusters)
    
    # Plot each cluster
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_labels == cluster_id
        
        if cluster_id == -1:  # Noise
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                      c='gray', s=50, alpha=0.3, 
                      label='Noise', marker='x')
        else:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                      c=[colors[cluster_id]], s=100, alpha=0.6,
                      label=f'Cluster {cluster_id}', edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_top_codes_heatmap(top_codes_dict: dict,
                           figsize: Tuple[int, int] = (12, 8),
                           title: str = "Top Codes per Cluster") -> plt.Figure:
    """
    Plot heatmap of top codes per cluster.
    
    Args:
        top_codes_dict: Dictionary from get_top_codes_per_cluster
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Prepare data for heatmap
    all_codes = set()
    for codes_list in top_codes_dict.values():
        all_codes.update([code for code, _ in codes_list])
    
    all_codes = sorted(all_codes)
    clusters = sorted(top_codes_dict.keys())
    
    # Build matrix
    matrix = np.zeros((len(all_codes), len(clusters)))
    
    for i, code in enumerate(all_codes):
        for j, cluster_id in enumerate(clusters):
            score = next((s for c, s in top_codes_dict[cluster_id] if c == code), 0)
            matrix[i, j] = score
    
    # Normalize by column (cluster)
    col_max = matrix.max(axis=0, keepdims=True)
    col_max[col_max == 0] = 1  # Avoid division by zero
    matrix_norm = matrix / col_max
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(matrix_norm, 
                xticklabels=[f"C{c}" for c in clusters],
                yticklabels=all_codes,
                cmap='YlOrRd',
                ax=ax,
                cbar_kws={'label': 'Normalized Score'})
    
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Medical Code')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_cluster_comparison(cluster_summary: pd.DataFrame,
                           figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
    """
    Plot comparison of cluster characteristics.
    
    Args:
        cluster_summary: DataFrame from get_cluster_summary()
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Filter out noise
    df = cluster_summary[~cluster_summary['is_noise']].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Number of patients
    axes[0].bar(df['cluster_id'], df['n_patients'], color='steelblue')
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Number of Patients')
    axes[0].set_title('Cluster Sizes')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Average codes per patient
    axes[1].bar(df['cluster_id'], df['avg_codes_per_patient'], color='coral')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Avg Codes per Patient')
    axes[1].set_title('Code Density')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Unique codes
    axes[2].bar(df['cluster_id'], df['unique_codes'], color='seagreen')
    axes[2].set_xlabel('Cluster')
    axes[2].set_ylabel('Number of Unique Codes')
    axes[2].set_title('Code Diversity')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def save_all_plots(pipeline, output_dir: str = 'plots'):
    """
    Generate and save all available plots for a fitted pipeline.
    
    Args:
        pipeline: Fitted FHIRClusteringPipeline
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Cluster sizes
    fig = plot_cluster_sizes(pipeline.get_cluster_labels())
    fig.savefig(f'{output_dir}/cluster_sizes.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir}/cluster_sizes.png")
    
    # Explained variance (if reduction was used)
    if pipeline.dim_reducer is not None:
        fig = plot_explained_variance(pipeline.dim_reducer.get_explained_variance())
        fig.savefig(f'{output_dir}/explained_variance.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_dir}/explained_variance.png")
    
    # 2D clusters
    if pipeline.reduced_data is not None:
        fig = plot_2d_clusters(pipeline.reduced_data, pipeline.get_cluster_labels())
        fig.savefig(f'{output_dir}/clusters_2d.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_dir}/clusters_2d.png")
    
    # Cluster comparison
    fig = plot_cluster_comparison(pipeline.get_cluster_summary())
    fig.savefig(f'{output_dir}/cluster_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir}/cluster_comparison.png")
    
    # Top codes heatmap
    top_codes = pipeline.get_top_codes_per_cluster(top_n=15)
    fig = plot_top_codes_heatmap(top_codes)
    fig.savefig(f'{output_dir}/top_codes_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir}/top_codes_heatmap.png")
    
    print(f"\nAll plots saved to {output_dir}/")
