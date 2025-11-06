# What to Watch: Graph-Based Movie Recommendation System

A movie recommendation system using heterogeneous graph neural networks (GNNs) to predict user preferences on the MovieLens 25M dataset. This project implements and compares multiple graph-based approaches including HinSAGE (Heterogeneous GraphSAGE) and Node2Vec for link prediction on user-movie bipartite graphs.

## Overview

Traditional recommendation systems often treat users and items as independent entities. This project leverages graph representation learning to capture the complex relationships between users and movies through a bipartite graph structure, where:
- **Nodes** represent users and movies
- **Edges** represent positive ratings (4+ stars)
- **Node features** include movie metadata (genres, premiere year, genome scores)

The graph neural network learns embeddings that encode both the structural patterns in the user-movie interaction graph and the content features of movies.

## Key Features

- **Heterogeneous Graph Neural Networks**: Implements HinSAGE for learning on bipartite user-movie graphs
- **Multiple GNN Approaches**: Includes HinSAGE, Node2Vec, and baseline comparisons
- **Rich Movie Features**: Incorporates genres, premiere years, and MovieLens genome scores (1,128 tag relevance scores)
- **Comprehensive Evaluation**: NDCG, Recall, and MRR metrics at multiple cutoffs (k=3, 5, 10)
- **Cold-Start Handling**: Separate evaluation pipeline for newly released movies
- **Temporal Splitting**: Train/validation/test split based on rating timestamps to simulate real-world deployment

## Dataset

The project uses the [MovieLens 25M dataset](https://grouplens.org/datasets/movielens/25m/), which contains:
- 25 million ratings from 162,000 users on 62,000 movies
- Movie metadata (genres, release years)
- Genome scores (relevance scores for 1,128 tags per movie)

### Data Processing Pipeline

1. **Filtering**: Keep active users (last rating in 2016+, 5-200 ratings) and popular movies (20+ ratings)
2. **Rating Classification**: Binary classification (positive ≥4 stars, negative ≤2.5 stars)
3. **Temporal Split**: 60% train / 20% validation / 20% test based on timestamps
4. **Cold-Start Split**: 8% of newest movies reserved for cold-start evaluation

**Final Dataset Statistics:**
- **Users**: 28,083
- **Movies**: 4,057 (warm) + 352 (cold-start)
- **Positive Ratings**: 1.5M (train: 896K, val: 291K, test: 325K)

## Model Architecture

### HinSAGE (Heterogeneous GraphSAGE)

The core model uses a two-layer HinSAGE architecture:

```
Layer 1: Aggregate 1-hop neighbors (15 samples per node)
Layer 2: Aggregate 2-hop neighbors (10 samples per node)
Hidden dimensions: [32, 32]
Link prediction: Inner product of user/movie embeddings → sigmoid
```

**Training Details:**
- Optimizer: Adam (lr=1e-2)
- Loss: Binary cross-entropy
- Batch size: 256 user-movie pairs
- Negative sampling: Hard negatives (observed poor ratings) + random unrated movies
- Early stopping: 5 epochs patience on validation loss

## Results

The HinSAGE model achieves strong performance on the test set with a 9:1 negative-to-positive ratio:

| Metric | @3 | @5 | @10 |
|--------|-----|-----|------|
| **NDCG** | 0.3803 | 0.3795 | 0.4174 |
| **Recall** | 0.162 | 0.249 | 0.4141 |
| **MRR** | 0.5391 | 0.5699 | 0.5835 |

- **Accuracy**: 69.7%
- **AUC**: 0.773

These results demonstrate the model's ability to rank truly relevant movies highly, with over 41% of actual positive ratings appearing in the top-10 recommendations.

## Project Structure

```
What-to-Watch/
├── code/
│   ├── HinSAGE.ipynb                  # Main HinSAGE implementation
│   ├── node2vec_v2.ipynb              # Node2Vec baseline
│   ├── multirelational_v1.ipynb       # Multi-relational experiments
│   ├── create_datasets.ipynb          # Data preprocessing pipeline
│   ├── explore_dataset.ipynb          # Exploratory data analysis
│   ├── evaluation_functions.py        # NDCG, Recall, MRR metrics
│   └── download_movielens_data.py     # Dataset download script
├── raw_data/                          # MovieLens 25M dataset
├── processed_data/                    # Train/val/test splits
├── models/                            # Trained model weights
├── environment_instructions.md        # Conda environment setup
└── grl_env.yml                       # Conda environment specification
```

## Tech Stack

- **Deep Learning**: TensorFlow 2.x, Keras
- **Graph ML**: StellarGraph (heterogeneous GNN library)
- **Data Processing**: Pandas, NumPy
- **Evaluation**: scikit-learn
- **Visualization**: Matplotlib

## Setup

### Prerequisites

- Python 3.8+
- Conda or Miniconda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/finbarrhodes/What-to-Watch.git
cd What-to-Watch
```

2. Create and activate the conda environment:
```bash
conda env create -f grl_env.yml
conda activate GRL_env
```

3. Download the MovieLens 25M dataset:
```bash
python code/download_movielens_data.py
```

4. Run the data preprocessing pipeline:
```bash
jupyter notebook code/create_datasets.ipynb
```

5. Train the HinSAGE model:
```bash
jupyter notebook code/HinSAGE.ipynb
```

## Usage

### Training a Model

The main HinSAGE model can be trained by running the `HinSAGE.ipynb` notebook, which:
1. Loads preprocessed user-movie graph data
2. Creates HinSAGE link generators for sampling subgraphs
3. Trains the model with early stopping
4. Evaluates on the test set with multiple metrics

### Evaluation

The evaluation pipeline computes ranking metrics for each user:
- **NDCG@k**: Normalized discounted cumulative gain
- **Recall@k**: Fraction of relevant items in top-k
- **MRR@k**: Mean reciprocal rank of first relevant item

Results are saved to `models/hinsage_results.json`.

## Key Implementation Details

### Heterogeneous Graph Construction

The bipartite graph is constructed using StellarGraph:
- **User nodes**: Single bias feature (constant=1)
- **Movie nodes**: 1,148 features (genres one-hot + genome scores + normalized year)
- **Edges**: Positive ratings only (negative ratings used for training sampling)

### Negative Sampling Strategy

For training and validation, negative samples are generated using a hybrid approach:
1. **Hard negatives**: Movies the user rated poorly (≤2.5 stars)
2. **Random negatives**: Unrated movies (if hard negatives insufficient)

This ensures the model learns to distinguish between truly liked movies and both disliked and unknown movies.

### Graph Sampling

HinSAGE uses neighborhood sampling to create mini-batches:
- Sample 15 neighbors at 1-hop distance
- Sample 10 neighbors at 2-hop distance
- This creates subgraphs with pattern: `user → movie → user` and `movie → user → movie`

## Future Improvements

- Implement attention mechanisms (e.g., GAT, HAN) for better neighbor aggregation
- Add temporal dynamics to capture evolving user preferences
- Incorporate social network information (user-user similarity)
- Explore multi-task learning with rating prediction and ranking
- Deploy as a REST API with real-time recommendations

## License

This project is available for educational and portfolio purposes.

## Acknowledgments

- MovieLens 25M dataset provided by [GroupLens Research](https://grouplens.org/)
- Built with [StellarGraph](https://stellargraph.readthedocs.io/) library
