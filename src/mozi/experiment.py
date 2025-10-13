from .training import Trainer, fetch_dataset, fetch_datasplitter, fetch_model
from .aggregation import aggregate
from .attack import attack
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from loguru import logger
from typing import Literal
import polars as pl
from datetime import datetime
import os
from sklearn.linear_model import Ridge
from math import ceil
import numpy as np
from scipy.stats import kurtosis

TEMPERATURE = 0.1
IMPORTANCE_TEMPERATURE = 0.1
NUM_SCORES = 2


@dataclass
class ExperimentConfig:
    exp_name: str
    model: str
    dataset: str
    split: str
    learning_rate: float
    n_client: int
    m_client: int
    n_server: int
    m_server: int
    device: str
    datapath: str
    batch_size: int
    n_epoch: int
    n_round: int
    num_workers: int
    attack: str
    aggregation: str
    selection_fraction: float
    method: Literal["stateful", "stateless", "baseline"]
    dir_alpha: float | None = None


class Experiment:
    def __init__(self, config: ExperimentConfig) -> None:
        train_set, self.test_set = fetch_dataset(config.datapath, config.dataset)

        self.train_subsets = fetch_datasplitter(
            train_set, config.split, config.n_client, alpha=config.dir_alpha
        ).split()

        self.n_epoch = config.n_epoch
        self.n_round = config.n_round
        self.attack = config.attack
        self.m_client = config.m_client
        self.n_client = config.n_client
        self.n_server = config.n_server
        self.m_server = config.m_server
        self.frac = config.selection_fraction
        self.method = config.method
        self.aggregation = config.aggregation
        self.config = config
        # --- 新增和修改的部分 ---
        self.record = []
        self.exp_name = config.exp_name

        self.credit = torch.zeros(self.n_client)
        self.ema_decay = 0.5
        self.history_for_regression = []
        self.max_history_size = 500
        self.credit_model = Ridge(alpha=1.0)
        self.retrain_interval = 2
        self.score_importances = torch.ones(NUM_SCORES) / NUM_SCORES
        self.prev_winner = 1
        # 定义结果文件的路径
        self.log_dir = "log"  # 可以自定义日志目录
        os.makedirs(self.log_dir, exist_ok=True)  # 确保目录存在
        self.output_file = os.path.join(self.log_dir, f"{self.exp_name}.parquet")

        # 实验开始时，可以选择性地删除旧文件，以保证每次运行都是全新的结果
        # 如果您希望追加到旧文件中，请注释掉下面这行
        if os.path.exists(self.output_file):
            logger.warning(
                f"Output file {self.output_file} already exists. Removing it for a fresh start."
            )
            os.remove(self.output_file)

        self.reset(config)

    @staticmethod
    def cos_sim_mat(X: torch.Tensor, Y: torch.Tensor):
        X_norm = F.normalize(X, dim=1)
        Y_norm = F.normalize(Y, dim=1)
        return X_norm.matmul(Y_norm.T)

    @staticmethod
    def _calc_kurtosis(x: torch.Tensor):
        x_np = x.cpu().numpy()
        return torch.from_numpy(kurtosis(x_np, axis=1, fisher=False)).to(x.device)

    def reset(self, config: ExperimentConfig):
        init_model = fetch_model(config.model)

        self.local_models = [
            fetch_model(config.model).to(config.device) for _ in range(config.n_client)
        ]

        self.clients = [
            Trainer(
                model=self.local_models[i],
                init_state=init_model.state_dict(),
                train_set=self.train_subsets[i],
                test_set=self.test_set,
                bs=config.batch_size,
                nw=config.num_workers,
                lr=config.learning_rate,
                device=config.device,
            )
            for i in range(config.n_client)
        ]

    def save_results(self, result: dict):
        # 将字典转换为 Polars DataFrame
        results_df = pl.DataFrame(result)

        try:
            # FileLock 依然非常重要，因为它能防止在“读-改-写”过程中发生竞态条件
            # with FileLock(f'{self.output_file}.lock', timeout=30): # 建议对锁文件使用不同扩展名
            if os.path.exists(self.output_file):
                # --- 正确的追加逻辑 ---
                # 1. 读取旧数据
                existing_df = pl.read_parquet(self.output_file)
                # 2. 将新旧数据垂直合并
                combined_df = pl.concat([existing_df, results_df], how="vertical")
                # 3. 将合并后的完整数据写回，覆盖原文件
                combined_df.write_parquet(self.output_file)
                logger.success(f"📈 Appended 1 record to {self.output_file}")
            else:
                # --- 文件首次创建的逻辑 ---
                # 直接调用 DataFrame 的 write_parquet 方法
                results_df.write_parquet(self.output_file)
                logger.success(f"💾 Saved initial record to {self.output_file}")

        except Exception as e:
            logger.error(
                f"❌ Failed to save results to Parquet file {self.output_file}: {e}"
            )

    def run(self):
        for r in range(self.n_round):
            if self.method == "stateless":
                loss, acc = self.mozi_fl(r, stateful=False)
            elif self.method == "stateful":
                loss, acc = self.mozi_fl(r, stateful=True)
            else:
                loss, acc = self.classic_fl(r)

            record = {
                "exp_name": self.exp_name,
                "timestamp": datetime.now().isoformat(),
                "loss": loss,
                "acc": acc,
                "rnd": r,
            }

            self.save_results(record)

    def classic_fl(self, r: int):
        logger.info(f"Round {r}: Start Training")
        for client in self.clients:
            client.local_train(self.n_epoch)
        logger.info(f"Round {r}: Training End.")

        client_updates = torch.stack([client.get_grad() for client in self.clients])
        client_updates = attack(
            client_updates, self.attack, self.m_client, self.n_client
        )

        server_updates = []
        for i in range(self.n_server):
            if i < self.m_server:
                # when the server is malicious
                update = aggregate(client_updates, "collude", m=self.m_client)
            else:
                # when the server is benign
                update = aggregate(client_updates, "fedavg")
            server_updates.append(update)
        server_updates = torch.stack(server_updates)

        logger.info(f"Round {r}: Aggregation End")
        global_update = aggregate(server_updates, self.aggregation, prop=0.8)

        for client in self.clients:
            client.set_grad(global_update)

        loss, acc = self.clients[-1].test()
        logger.success(f"Round {r}: Loss: {loss:.4f}, Acc: {acc * 100:.2f}!")
        return loss, acc

    def mozi_fl(self, r: int, stateful=False):
        # local training and simulating attacks
        logger.info(f"Round {r}: Start Training")
        for client in self.clients:
            client.local_train(self.n_epoch)
        logger.info(f"Round {r}: Training End.")
        client_updates = torch.stack([client.get_grad() for client in self.clients])
        client_updates = attack(
            client_updates, self.attack, self.m_client, self.n_client
        )

        # select client sbubsets
        selected_index = self._select_clients(
            num_selected=int(self.frac * self.n_client), temperature=0.3
        )
        logger.info(f"Round {r}: Server Selection:\n {selected_index}.")

        # aggregate and score
        server_updates = self._get_server_updates(client_updates, selected_index)

        scores = self._calc_scores(client_updates, server_updates)

        self._collect_regression_data(scores, selected_index)
        self._update_credits_if_needed(r)
        self._log_credit_stats()

        composite_scores = (scores * self.score_importances).sum(dim=1)

        # identify and broadcast the `winner` update
        winner = composite_scores.argmax()
        logger.success(f"Round {r}: Welcome our new winner: {winner.item()}!")
        global_update = server_updates[winner]
        self.prev_winner = winner
        for client in self.clients:
            client.set_grad(global_update)

        # test and log
        loss, acc = self.clients[-1].test()
        logger.success(f"Round {r}: Loss: {loss:.4f}, Acc: {acc * 100:.2f}!")
        return loss, acc

    def _select_clients(
        self, num_selected: int, temperature: float = 0.1
    ) -> torch.Tensor:
        """
        为每个服务器选择客户端。
        对于 prev_winner，使用 softmax 概率采样。
        对于其他服务器，使用随机采样。

        Args:
            num_selected (int): 要选择的客户端数量。
            temperature (float): Softmax 的温度参数。
        """
        return torch.stack(
            [
                torch.multinomial(
                    torch.softmax(self.credit, dim=0).cpu() / temperature,
                    num_samples=num_selected,
                    replacement=False,
                )
                for _ in range(self.n_server)
            ]
        )

    def _get_server_updates(
        self, client_updates: torch.Tensor, selected_index: torch.Tensor
    ) -> torch.Tensor:
        server_updates = []
        for i in range(self.n_server):
            if i < self.m_server:
                update = aggregate(client_updates, "collude", m=self.m_client)
            else:
                update = aggregate(client_updates[selected_index[i]], "fedavg")
            server_updates.append(update)
        server_updates = torch.stack(server_updates)

        return server_updates

    def _rescale_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """对单批次内的分数进行Min-Max缩放，使其分布在[0, 1]"""
        min_val = torch.min(scores)
        max_val = torch.max(scores)

        # 处理所有值都相同的边缘情况，避免除以零
        if max_val == min_val:
            # 可以返回全0.5或全0，取决于你的偏好
            return torch.full_like(scores, 0.5)

        return (scores - min_val) / (max_val - min_val)

    def _calc_scores(self, client_updates: torch.Tensor, server_updates: torch.Tensor):
        """calculate 3 socres between client and server updates"""

        # similarity scores
        cos_scores = self.cos_sim_mat(server_updates, client_updates)

        # magnitude scores
        server_norms = torch.norm(server_updates, p=2, dim=1).unsqueeze(1)
        client_norms = torch.norm(client_updates, p=2, dim=1).unsqueeze(0)
        mag_scores = 1 - torch.abs(client_norms - server_norms) / (
            client_norms + server_norms + 1e-9
        )

        # sign scores
        # server_signs = self._get_sign_stats(server_updates).unsqueeze(1)
        # client_signs = self._get_sign_stats(client_updates).unsqueeze(0)
        # sgn_scores = 1 - torch.abs(client_signs - server_signs)

        cos_scores = self._rescale_scores(cos_scores)
        mag_scores = self._rescale_scores(mag_scores)

        all_scores = torch.stack([cos_scores, mag_scores])
        median_scores, _ = all_scores.median(dim=2)
        logger.info(f"Round scores: {median_scores}")
        return median_scores.T.cpu()

    def _collect_regression_data(
        self, scores: torch.Tensor, selected_index: torch.Tensor
    ):
        """
        Identifies a set of trusted servers for the current round, and only adds
        their participation data and standardized features to the history.
        """
        # probe_features is a (K, d) tensor
        num_probes, num_dims = scores.shape

        # --- 1. 计算临时的综合分数以识别可信服务器 ---
        # 使用与 mozi_fl 中相同的权重来确保一致性
        composite_scores = (scores * self.score_importances).sum(dim=1)

        # --- 2. 识别本轮的可信服务器集 (您的原始逻辑) ---
        # 根据综合分数进行排序
        _, sorted_indices = torch.sort(composite_scores, descending=True)

        # 选择分数最高的 top 50% (或至少1个) 作为可信集
        num_trusted_servers = max(1, ceil(self.n_server / 2))
        trusted_server_indices = sorted_indices[:num_trusted_servers]

        logger.info(
            f"Trusted server set for data collection: {trusted_server_indices.tolist()}"
        )

        if trusted_server_indices.numel() == 0:
            logger.warning("No trusted servers identified. Skipping data collection.")
            return

        # 在这个“干净”的数据集上计算均值和标准差
        mean = scores.mean(dim=0)
        std = scores.std(dim=0).clamp(min=1e-6)

        # 标准化所有探针的特征，但使用可信集的统计数据作为基准
        # 这样，即使是“坏”探针，其分数也会被转换到这个“好”的坐标系下
        standardized_features = (scores - mean) / std

        # --- 4. 为本轮的可信服务器创建并添加数据点 ---
        new_data_points = []
        # 只遍历可信服务器的索引
        for k in trusted_server_indices:
            participation_vector = np.zeros(self.n_client)
            selected_clients = selected_index[k]
            participation_vector[selected_clients.cpu().numpy()] = 1

            # 目标是标准化的多维特征向量
            target_vector = standardized_features[k].cpu().numpy()

            new_data_points.append((participation_vector, target_vector))
            logger.debug(
                f"Adding to history: P-Vec (sum={participation_vector.sum()}), Target={np.round(target_vector, 2)}"
            )

        # --- 5. 追加到历史记录并管理大小 (逻辑不变) ---
        self.history_for_regression.extend(new_data_points)
        if len(self.history_for_regression) > self.max_history_size:
            self.history_for_regression = self.history_for_regression[
                -self.max_history_size :
            ]

    def _update_credits_if_needed(self, r: int):
        """检查是否到达再训练周期，如果满足条件，则触发信誉模型的再训练。"""
        if (
            r > 0
            and r % self.retrain_interval == 0
            and len(self.history_for_regression) > self.n_client
        ):
            logger.info(f"Round {r}: Retraining credit model")
            self.update_credit_with_regression()

    def _log_credit_stats(self):
        """打印当前良性与恶意客户端的平均信誉统计信息。"""
        benign_avg_credit = self.credit[self.m_client :].mean()
        malicious_avg_credit = self.credit[: self.m_client].mean()
        logger.info(f"Benign: {benign_avg_credit:.4f}, Mal: {malicious_avg_credit:.4f}")

    @staticmethod
    def _get_sign_stats(tensor: torch.Tensor) -> torch.Tensor:
        """Calculates the sign statistics (non-negative counts) for a tensor."""
        return (tensor >= 0).float().sum(dim=1) / tensor.shape[1]

    def update_credit_with_regression(self):
        """
        Uses historical data to train a multi-target regression model and updates client credits.
        """
        if (
            not self.history_for_regression
            or len(self.history_for_regression) < self.n_client
        ):
            logger.warning("Not enough history for regression. Skipping credit update.")
            return

        # 1. 准备训练数据
        X_train = np.array(
            [item[0] for item in self.history_for_regression]
        )  # Shape: (T_history, N)
        y_train = np.array(
            [item[1] for item in self.history_for_regression]
        )  # Shape: (T_history, d)

        # 检查y_train中是否有NaN值 (可能由std=0导致)
        if np.isnan(y_train).any():
            logger.warning("NaNs found in y_train, filling with 0.")
            y_train = np.nan_to_num(y_train)

        sample_importance = np.linalg.norm(y_train, axis=1) + 0.1
        try:
            # scikit-learn's Ridge seamlessly handles a 2D y_train
            self.credit_model.fit(X_train, y_train, sample_weight=sample_importance)

            # model.coef_ shape will be (d, N), so we transpose it
            new_credits_matrix = torch.from_numpy(
                self.credit_model.coef_.T
            ).float()  # Shape: (N, d)
            # 简单求和即可，因为岭回归的系数已经反映了每个特征的重要性
            new_credits_vector = new_credits_matrix.sum(dim=1)  # Shape: (N,)
            self.credit = new_credits_vector

            logger.success("Multi-target credit model retrained. Credits updated.")
            logger.info(
                f"Learned credit matrix :\n{new_credits_matrix}"
            )
            logger.info(
                f"New credit:\n{new_credits_vector}"
            )

        except Exception as e:
            logger.error(f"Failed to train credit model: {e}")
            logger.error(
                f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}"
            )
