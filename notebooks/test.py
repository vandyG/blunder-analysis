# %%
import chess.pgn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
from polars import selectors as cs


# %%
input_dir = "/home/vandy/Work/MATH6310/blunder-analysis/data/gold"
my_games = pl.scan_parquet(input_dir)

# %%
my_games.collect_schema()

# %%
important_columns = [
    "game_time",
    "increment",
    "turn",
    "cp_score",
    "winning_chance",
    "drawing_chance",
    "losing_chance",
    "piece_type",
    "is_check",
    "mate_in",
    "clock",
    "eval_delta",
    "judgement",
    "time_control_type",
    "player_elo",
    "time_ratio",
    "total_chance",
    "fen_tensor",
]
important_columns = list(dict.fromkeys(important_columns))
raw_lazy = my_games.select(important_columns)

raw_pl_df = raw_lazy.with_columns(cs.by_dtype(pl.Boolean).cast(pl.Int64))
print(raw_pl_df.schema)

# # %%
# categorical_cols = ["piece_type", "judgement", "time_control_type"]
# boolean_cols = ["turn", "is_check"]
# numeric_cols = [
#     "game_time",
#     "increment",
#     "cp_score",
#     "winning_chance",
#     "drawing_chance",
#     "losing_chance",
#     "mate_in",
#     "clock",
#     "eval_delta",
#     "player_elo",
#     "time_ratio",
#     "total_chance",
# ]
# required_cols = numeric_cols + categorical_cols + boolean_cols
# clean_df = (
#     raw_df.dropna(subset=required_cols + ["fen_tensor"]).reset_index(drop=True).copy()
# )
# fen_matrix = np.vstack(clean_df.pop("fen_tensor").to_numpy()).astype(np.float32)
# for col in boolean_cols:
#     clean_df[col] = clean_df[col].astype(int)
# tabular_encoded = pd.get_dummies(
#     clean_df,
#     columns=categorical_cols,
#     dtype=np.float32,
# )
# numeric_matrix = tabular_encoded.to_numpy(dtype=np.float32)
# X_sparse = sp.hstack(
#     [
#         sp.csr_matrix(numeric_matrix),
#         sp.csr_matrix(fen_matrix),
#     ],
#     format="csr",
#     dtype=np.float32,
# )
# scaler = StandardScaler(with_mean=False)
# X_scaled = scaler.fit_transform(X_sparse)
# print(f"Encoded tabular shape: {numeric_matrix.shape}")
# print(f"FEN matrix shape: {fen_matrix.shape}")
# print(f"Sparse feature matrix: {X_scaled.shape}")

# # %%
# n_components = 20
# svd = TruncatedSVD(n_components=n_components, random_state=0)
# svd_embedding = svd.fit_transform(X_scaled)
# explained_df = pd.DataFrame({
#     "component": np.arange(1, n_components + 1),
#     "explained_variance_ratio": svd.explained_variance_ratio_,
#     "cumulative_explained_variance": np.cumsum(svd.explained_variance_ratio_),
# })
# explained_df

# # %%
# fig, ax = plt.subplots(figsize=(8, 4))
# ax.bar(
#     explained_df["component"],
#     explained_df["explained_variance_ratio"],
#     color="#1f77b4",
#     label="Individual",
# )
# ax.plot(
#     explained_df["component"],
#     explained_df["cumulative_explained_variance"],
#     color="#ff7f0e",
#     marker="o",
#     label="Cumulative",
# )
# ax.set_xlabel("Component")
# ax.set_ylabel("Explained variance ratio")
# ax.set_title("TruncatedSVD variance explained")
# ax.set_xticks(explained_df["component"])
# ax.set_ylim(0, 1)
# ax.legend()
# plt.tight_layout()

# # %% [markdown]
# # ## Why TruncatedSVD instead of PCA
# # 
# # PCA requires centering every feature, which would densify the sparse design matrix built from the 768-dimensional `fen_tensor` and explode memory usage.
# # 
# # TruncatedSVD operates directly on sparse inputs without explicit centering, giving the same principal directions for zero-mean data while remaining computationally feasible.


