from .training import Trainer, fetch_dataset, fetch_datasplitter, fetch_model
from .aggregation import aggregate
from .attack import attack
from dataclasses import dataclass, field, asdict
import torch
import torch.nn.functional as F
from loguru import logger
from typing import Literal
import petname
import polars as pl
@dataclass
class ExperimentConfig:
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
    method: Literal['ours', 'baseline']
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
        self.petname = petname.generate()
        self.config = config
        self.reset(config)
        self.record = []

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
        
    def save_results(self):
        """
        将收集到的所有性能记录与实验配置合并，
        转换为一个扁平的 DataFrame 并保存。
        """
        logger.info("Experiment finished. Saving results...")
        
        if not self.record:
            logger.warning("No performance records to save.")
            return

        # 1. 将每轮的性能记录转换为 DataFrame
        perf_df = pl.DataFrame(self.record)
        
        # 2. 将 ExperimentConfig 转换为字典
        #    asdict() 会返回一个普通的 Python 字典
        config_dict = asdict(self.config)
        
        # --- 优雅地合并配置信息的关键步骤 ---
        
        # 3. 使用 with_columns 将配置字典中的每个键值对添加为新列
        #    pl.lit(value) 创建一个字面量（常量）列。
        #    Polars 会自动将其广播到所有行。
        results_df = perf_df.with_columns([
            pl.lit(value).alias(key) for key, value in config_dict.items()
        ]).with_columns([
            pl.lit(self.petname).alias('petname')
        ])
        
        
        
        # (可选但推荐) 重新排序列的顺序，让关键信息更靠前
        # 将性能指标放在前面，配置信息放在后面
        fixed_cols = ['petname', 'rnd', 'loss', 'acc']
        # 自动获取所有其他配置列
        config_cols = [col for col in results_df.columns if col not in fixed_cols]
        
        results_df = results_df.select(fixed_cols + sorted(config_cols))

        print(results_df)
        # 4. 定义输出路径
        #    现在我们只需要一个文件，所以文件名可以更通用


        # 5. 写入单个 Parquet 文件
        results_df.write_parquet('record.parquet')
        
        logger.success("Combined results and config saved.'")
        
    def run(self):
        for r in range(self.n_round):
            if self.method == 'ours':
                loss, acc = self.boomerang_fl(r)
            else:
                loss, acc = self.classic_fl(r)
                
            self.record.append({
                'loss': loss,
                'acc': acc,
                'rnd': r,
            })
            
            self.save_results()
        

    def classic_fl(self, r: int):
        logger.info(f'Round {r}: Start Training')
        for client in self.clients:
            client.local_train(self.n_epoch)
        logger.info(f'Round {r}: Training End.')
        
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
        
        logger.info(f'Round {r}: Aggregation End')
        global_update = server_updates.flatten()
        
        for client in self.clients:
            client.set_grad(global_update)
            
        loss, acc = self.clients[-1].test()
        logger.success(f'Round {r}: Loss: {loss:.4f}, Acc: {acc*100:.2f}!')
        return loss, acc

    def boomerang_fl(self, r: int):
        logger.info(f'Round {r}: Start Training')
        for client in self.clients:
            client.local_train(self.n_epoch)
        logger.info(f'Round {r}: Training End.')
        
        client_updates = torch.stack([client.get_grad() for client in self.clients])
        client_updates = attack(
            client_updates, self.attack, self.m_client, self.n_client
        )

        # client selection based on credit-probability
        selected_index = torch.stack(
            [
                torch.randperm(self.n_client)[:int(self.n_client * self.frac)]
                for _ in range(self.n_server)
            ]
        )
        logger.info(f'Round {r}: Server Selection: {selected_index}.')

        server_updates = []
        for i in range(self.n_server):
            if i < self.m_server:
                # when the server is malicious
                update = aggregate(client_updates, "collude", m=self.m_client)
            else:
                # when the server is benign
                update = aggregate(client_updates[selected_index[i]], "fedavg")
            server_updates.append(update)
        server_updates = torch.stack(server_updates)
        
        logger.info(f'Round {r}: Aggregation End')

        # similarity calculation & global update confirm
        similarities = self.cos_sim_mat(server_updates, client_updates)
        logger.info(f'Round {r}: Scores: {similarities}.')
        scores, _ = similarities.median(dim=1)
        logger.info(f'Round {r}: Server Scores: {scores}.')
        # winner = torch.abs(scores - scores.median()).argmin()
        winner = scores.argmax()
        logger.success(f'Round {r}: Welcome our new winner: {winner.item()}!')
        # diff = torch.abs(scores - scores.median(dim=0)[0])
        # winner = diff.argmin()
        global_update = server_updates[winner]
        
        for client in self.clients:
            client.set_grad(global_update)
            
        loss, acc = self.clients[-1].test()
        logger.success(f'Round {r}: Loss: {loss:.4f}, Acc: {acc*100:.2f}!')
        return loss, acc
