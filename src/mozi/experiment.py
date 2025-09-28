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
from sklearn.linear_model import Ridge, ElasticNet
from math import ceil
import numpy as np

TEMPERETURE = 0.1


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
    method: Literal["ours", "baseline"]
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
        self.retrain_interval = 5

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
        # ... (1. 训练与攻击部分不变) ...
        logger.info(f"Round {r}: Start Training")
        for client in self.clients:
            client.local_train(self.n_epoch)
        logger.info(f"Round {r}: Training End.")
        client_updates = torch.stack([client.get_grad() for client in self.clients])
        client_updates = attack(client_updates, self.attack, self.m_client, self.n_client)

        # --- 2. 客户端选择 ---
        selected_index = self._get_selection_indices(r, stateful)
        logger.info(f"Round {r}: Server Selection:\n {selected_index}.")

        # --- 3. 服务器聚合与评分 ---
        server_updates, scores = self._get_server_updates_and_scores(client_updates, selected_index)

        # --- 4. 信誉系统管理 (仅在 stateful 模式下) ---
        if stateful:
            # 4a. 收集本轮的新证据
            self._collect_regression_data(scores, selected_index)
            
            # 4b. 根据证据，在需要时更新我们的信念
            self._update_credits_if_needed(r)
            
            # 4c. 记录并报告当前的信念状态
            self._log_credit_stats()

        # --- 5. 确定并分发全局更新 ---
        winner = scores.argmax()
        logger.success(f"Round {r}: Welcome our new winner: {winner.item()}!")
        global_update = server_updates[winner]

        for client in self.clients:
            client.set_grad(global_update)

        # --- 6. 测试与返回结果 ---
        loss, acc = self.clients[-1].test()
        logger.success(f"Round {r}: Loss: {loss:.4f}, Acc: {acc * 100:.2f}!")
        return loss, acc
    
    def _get_selection_indices(self, r: int, stateful: bool) -> torch.Tensor:
        """根据模式（stateful/stateless）生成客户端选择索引。"""
        num_selected = int(round(self.n_client * self.frac))

        if stateful:
            # 完全复刻你代码中的“平移信誉”策略
            scaled_credit = self.credit.cpu() / TEMPERETURE
            min_credit = scaled_credit.min()
            epsilon = 1e-4
            shifted_credit = scaled_credit - min_credit + epsilon
            credit_prob = shifted_credit / shifted_credit.sum()
            
            logger.info("credit probabilities (sample)", credit_prob)

            if torch.isnan(credit_prob).any():
                logger.warning("Credit probabilities contained NaN. Falling back to uniform selection.")
                credit_prob = torch.ones(self.n_client) / self.n_client

            return torch.stack(
                [
                    torch.multinomial(credit_prob, num_selected, replacement=False)
                    for _ in range(self.n_server)
                ]
            )
        else:
            # Stateless 模式
            return torch.stack(
                [
                    torch.randperm(self.n_client)[:num_selected]
                    for _ in range(self.n_server)
                ]
            )
            
    def _get_server_updates_and_scores(self, client_updates: torch.Tensor, selected_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """聚合服务器更新并计算它们的分数。"""
        server_updates = []
        for i in range(self.n_server):
            if i < self.m_server:
                update = aggregate(client_updates, "collude", m=self.m_client)
            else:
                update = aggregate(client_updates[selected_index[i]], "fedavg")
            server_updates.append(update)
        server_updates = torch.stack(server_updates)

        similarities = self.cos_sim_mat(server_updates, client_updates)
        scores, _ = similarities.median(dim=1)
        scores = scores.cpu().squeeze()
        
        logger.info(f"Round scores: {scores}")
        return server_updates, scores
    
    def _collect_regression_data(self, scores: torch.Tensor, selected_index: torch.Tensor):
        """
        处理当前轮次的数据，并将其作为新的训练样本添加到多轮历史记录中。
        使用EMA平滑的统计数据进行标准化，以保证跨轮次的可比性。
        """
        # 1. 识别本轮的可信服务器集
        _, sorted_indices = torch.sort(scores)
        num_trusted_servers = max(1, ceil(self.n_server / 2))
        trusted_server_indices = sorted_indices[-num_trusted_servers:]
        logger.info(f"Trusted server set for data collection: {trusted_server_indices.tolist()}")

        # 2. 计算本轮的临时统计数据
        if trusted_server_indices.numel() == 0:
            logger.warning("No trusted scores found, skipping data collection for this round.")
            return
            
        current_mean = scores.mean().item()
        current_std = scores.std().item()
        stable_std = max(current_std, 1e-6)


        # 4. 为本轮的可信服务器创建标准化后的数据点
        new_data_points = []
        for i in trusted_server_indices:
            participation_vector = np.zeros(self.n_client)
            selected_clients = selected_index[i]
            participation_vector[selected_clients.cpu().numpy()] = 1
            
            raw_score = scores[i].item()
            standardized_score = (raw_score - current_mean) / stable_std
            logger.info(f'regression history add: {participation_vector.tolist()}, {standardized_score}')
            new_data_points.append((participation_vector, standardized_score))

        # 5. 将本轮的新数据点追加到多轮历史记录中
        self.history_for_regression.extend(new_data_points)

        # 6. 管理历史数据大小
        if len(self.history_for_regression) > self.max_history_size:
            self.history_for_regression = self.history_for_regression[-self.max_history_size:]

    def _update_credits_if_needed(self, r: int):
        """检查是否到达再训练周期，如果满足条件，则触发信誉模型的再训练。"""
        if r > 0 and r % self.retrain_interval == 0 and len(self.history_for_regression) > self.n_client:
            logger.info(f"Round {r}: Retraining credit model")
            self.update_credit_with_regression()

    def _log_credit_stats(self):
        """打印当前良性与恶意客户端的平均信誉统计信息。"""
        benign_avg_credit = self.credit[self.m_client:].mean()
        malicious_avg_credit = self.credit[:self.m_client].mean()
        logger.info(
            f"Benign: {benign_avg_credit:.4f}, Mal: {malicious_avg_credit:.4f}"
        )
    
    def update_credit_with_regression(self):
        """
        使用历史数据和样本权重训练回归模型，并用EMA更新客户端信誉。
        """
        if not self.history_for_regression:
            logger.warning("Regression history is empty. Skipping credit update.")
            return

        # 准备训练数据
        X_train = np.array([item[0] for item in self.history_for_regression])
        y_train = np.array([item[1] for item in self.history_for_regression])
        
        # --- 方案一：创建样本权重 ---
        # 使用标准化得分的绝对值作为样本权重。
        # 得分越偏离平均值（无论是极好还是极坏），该样本在训练中的重要性就越高。
        # 加上一个很小的常数，以确保即使得分为0的样本也有一定的权重。
        sample_weights = np.abs(y_train) + 0.1
        # --- 修改结束 ---

        try:
            # --- 方案一：将样本权重传递给 .fit() 方法 ---
            self.credit_model.fit(X_train, y_train, sample_weight=sample_weights)
            
            # 获取模型权重作为新的信誉分数
            new_credits = torch.from_numpy(self.credit_model.coef_).float()
            print(new_credits)
            # 使用EMA平滑更新，这是实现“逐步”边缘化的关键
            self.credit = self.ema_decay * self.credit.cpu() + \
                          (1 - self.ema_decay) * new_credits
            
            # 不使用EMA
            # self.credit = new_credits
                          
            logger.success("Credit model retrained and credits updated successfully.")

        except Exception as e:
            logger.error(f"Failed to train credit model: {e}")
            # 可以在这里加入调试信息，例如打印 X_train.shape, y_train.shape
            logger.error(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
