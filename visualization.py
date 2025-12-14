import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE


def visualize_patch_features(patch_list):
    batch_idx = 0
    flattened_features = []
    labels = []

    for patch_idx, patch in enumerate(patch_list):
        # patch shape: (B, 6, N, C) or similar
        p = patch[batch_idx]  # (6, N, C)
        _, N, C = p.shape

        p_flat = p.detach().cpu().reshape(-1, C).numpy()
        flattened_features.append(p_flat)
        labels.extend([patch_idx] * len(p_flat))

    features_all = np.concatenate(flattened_features, axis=0)
    labels = np.array(labels)

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30
    )
    features_2d = tsne.fit_transform(features_all)

    # Visualization settings
    sns.set(style="whitegrid", font_scale=1.2)

    num_patches = len(patch_list)
    palette = sns.color_palette("husl", num_patches)

    plt.figure(figsize=(8, 6))

    for i in range(num_patches):
        plt.scatter(
            features_2d[labels == i, 0],
            features_2d[labels == i, 1],
            s=15,
            color=palette[i],
            label=f"Patch {i + 1}",
            alpha=0.7,
            edgecolors="black",
            linewidths=0.3
        )

    plt.title(
        "t-SNE Clustering of Patch Features",
        fontsize=16,
        weight="bold"
    )
    plt.xlabel("t-SNE Dim 1", fontsize=12)
    plt.ylabel("t-SNE Dim 2", fontsize=12)

    plt.legend(
        loc="upper right",
        bbox_to_anchor=(1.15, 1),
        frameon=False
    )

    plt.tight_layout()
    plt.show()






def reduce_to_2d(embedding: torch.Tensor, method='pca'):
    embedding_np = embedding.detach().cpu().numpy()
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=5, random_state=42)
    else:
        raise NotImplementedError("Only 'pca' and 'tsne' are supported.")
    return reducer.fit_transform(embedding_np)

def plot_embedding(ax, embedding_2d, title, color='blue'):
    ax.scatter(
        embedding_2d[:, 0], embedding_2d[:, 1],
        c=color,
        s=30,  # ✅ 减小点的大小（推荐 15~30）
        edgecolors='k',
        alpha=0.7
    )
    ax.set_title(title, fontsize=9)  # ✅ 子图标题字体减小
    ax.grid(True)
    ax.set_xticks([])
    ax.set_yticks([])

def visualize_all_node_embeddings(node_forward1, node_backward1, node_forward2, node_backward2, method='pca'):


    node_forward2_mean = node_forward2[:, -1].mean(dim=0)
    node_backward2_mean = node_backward2[:, -1].mean(dim=0)

    # 降维
    emb_fwd1_2d = reduce_to_2d(node_forward1, method)
    emb_bwd1_2d = reduce_to_2d(node_backward1, method)
    emb_fwd2_2d = reduce_to_2d(node_forward2_mean, method)
    emb_bwd2_2d = reduce_to_2d(node_backward2_mean, method)

    # 可视化布局
    fig, axs = plt.subplots(2, 2, figsize=(6, 5))

    plot_embedding(axs[0, 0], emb_fwd1_2d, "Forward Predefined Graph Embedding", color='#007ACC')  # 深蓝
    plot_embedding(axs[0, 1], emb_bwd1_2d, "Backward Predefined Graph Embedding", color='#007ACC')
    plot_embedding(axs[1, 0], emb_fwd2_2d, "Forward Learned Graph Embedding", color='#00B3B3')  # 青色
    plot_embedding(axs[1, 1], emb_bwd2_2d, "Backward Learned Graph Embedding", color='#00B3B3')

    plt.tight_layout()
    plt.savefig("graph64.pdf", format='pdf')  # 保存为 PDF 文件
    plt.show()