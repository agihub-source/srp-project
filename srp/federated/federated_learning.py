"""联邦学习模块"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..monitoring import monitoring
from ..network.p2p import P2PNetwork
from ..security.encryption import EncryptionAlgorithm, SecurityManager

logger = logging.getLogger(__name__)

class AggregationStrategy(Enum):
    """聚合策略"""
    FEDAVG = auto()     # 联邦平均
    WEIGHTED = auto()    # 加权平均
    MEDIAN = auto()      # 中位数聚合
    TRIMMED = auto()    # 修剪平均
    KRUM = auto()       # Krum聚合

@dataclass
class ModelUpdate:
    """模型更新"""
    node_id: str                # 节点ID
    round_id: int              # 轮次ID
    params: Dict[str, Any]     # 模型参数
    metrics: Dict[str, float]  # 评估指标
    weight: float = 1.0        # 权重
    timestamp: float = field(default_factory=time.time)  # 时间戳

@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 32       # 批次大小
    epochs: int = 1           # 轮数
    learning_rate: float = 0.01  # 学习率
    optimizer: str = "adam"    # 优化器
    loss_fn: str = "mse"      # 损失函数
    device: str = "cpu"       # 设备
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])  # 评估指标

class FederatedLearner:
    """联邦学习器"""
    
    def __init__(
        self,
        network: P2PNetwork,
        security: SecurityManager,
        model: nn.Module,
        train_data: DataLoader,
        val_data: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        strategy: AggregationStrategy = AggregationStrategy.FEDAVG,
        min_clients: int = 2,
        max_wait_time: int = 60,
        encryption_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.RSA
    ):
        self.network = network
        self.security = security
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config or TrainingConfig()
        self.strategy = strategy
        self.min_clients = min_clients
        self.max_wait_time = max_wait_time
        self.encryption_algorithm = encryption_algorithm
        
        # 训练状态
        self.current_round = 0
        self.best_metrics = {}
        self._pending_updates: Dict[str, ModelUpdate] = {}
        
        # 生成密钥对
        self.key_pair = self.security.generate_key_pair(
            self.encryption_algorithm
        )
        
        # 注册消息处理器
        self.network.register_message_handler(
            "model_update",
            self._handle_model_update
        )
        self.network.register_message_handler(
            "global_model",
            self._handle_global_model
        )
        
        # 监控指标
        self.training_counter = monitoring.counter(
            "srp_federated_training_rounds_total",
            "Total number of federated training rounds",
            ["role"]
        )
        
        self.update_counter = monitoring.counter(
            "srp_model_updates_total",
            "Total number of model updates",
            ["status"]
        )
        
        self.metric_gauge = monitoring.gauge(
            "srp_training_metrics",
            "Training metrics",
            ["metric"]
        )
        
        self.timing_histogram = monitoring.histogram(
            "srp_training_time_seconds",
            "Training time in seconds",
            ["phase"]
        )
        
    async def start_training(
        self,
        num_rounds: int,
        **kwargs
    ) -> Dict[str, float]:
        """
        开始训练
        :param num_rounds: 训练轮数
        :param kwargs: 附加参数
        :return: 训练结果
        """
        try:
            # 广播训练开始
            await self.network.broadcast(
                "training_start",
                {
                    "num_rounds": num_rounds,
                    "config": self.config.__dict__,
                    "public_key": self.key_pair.public_key.hex(),
                    **kwargs
                }
            )
            
            for round_id in range(num_rounds):
                self.current_round = round_id
                
                # 本地训练
                local_update = await self._train_local()
                
                # 等待其他节点的更新
                updates = await self._collect_updates()
                
                # 聚合模型
                if len(updates) >= self.min_clients:
                    global_model = self._aggregate_updates(updates)
                    
                    # 更新本地模型
                    self._update_model(global_model)
                    
                    # 广播全局模型
                    await self._broadcast_global_model(global_model)
                    
                    # 评估模型
                    metrics = await self._evaluate_model()
                    
                    # 更新最佳指标
                    self._update_best_metrics(metrics)
                    
                    # 更新监控指标
                    for name, value in metrics.items():
                        self.metric_gauge.set(
                            value,
                            labels={"metric": name}
                        )
                        
                    logger.info(
                        f"Round {round_id + 1}/{num_rounds} "
                        f"completed with metrics: {metrics}"
                    )
                    
                else:
                    logger.warning(
                        f"Insufficient updates received in round {round_id + 1}"
                    )
                    
            return self.best_metrics
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
            
    async def _train_local(self) -> ModelUpdate:
        """
        本地训练
        :return: 模型更新
        """
        start_time = time.time()
        
        try:
            # 设置模型为训练模式
            self.model.train()
            device = torch.device(self.config.device)
            self.model.to(device)
            
            # 创建优化器
            optimizer = self._create_optimizer()
            
            # 创建损失函数
            loss_fn = self._create_loss_function()
            
            # 训练过程
            total_loss = 0.0
            num_batches = 0
            
            for epoch in range(self.config.epochs):
                epoch_loss = 0.0
                
                for batch in self.train_data:
                    # 移动数据到设备
                    inputs, targets = self._prepare_batch(batch, device)
                    
                    # 前向传播
                    outputs = self.model(inputs)
                    loss = loss_fn(outputs, targets)
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # 累计损失
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                total_loss += epoch_loss
                
            # 计算平均损失
            avg_loss = total_loss / (self.config.epochs * num_batches)
            
            # 获取模型参数
            params = {
                name: param.data.cpu().numpy()
                for name, param in self.model.named_parameters()
            }
            
            # 加密参数
            encrypted_params = self._encrypt_params(params)
            
            # 创建更新
            update = ModelUpdate(
                node_id=self.network.node_id,
                round_id=self.current_round,
                params=encrypted_params,
                metrics={"loss": avg_loss},
                timestamp=time.time()
            )
            
            # 更新监控指标
            duration = time.time() - start_time
            self.timing_histogram.observe(
                duration,
                labels={"phase": "local_training"}
            )
            
            return update
            
        except Exception as e:
            logger.error(f"Local training failed: {e}")
            raise
            
    async def _collect_updates(
        self,
        timeout: Optional[float] = None
    ) -> List[ModelUpdate]:
        """
        收集更新
        :param timeout: 超时时间
        :return: 更新列表
        """
        start_time = time.time()
        timeout = timeout or self.max_wait_time
        updates = []
        
        try:
            while len(updates) < self.min_clients:
                # 检查超时
                if time.time() - start_time > timeout:
                    break
                    
                # 等待更新
                await asyncio.sleep(1)
                
                # 收集已完成的更新
                current_updates = [
                    update
                    for update in self._pending_updates.values()
                    if update.round_id == self.current_round
                ]
                
                if current_updates:
                    updates.extend(current_updates)
                    for update in current_updates:
                        self._pending_updates.pop(update.node_id)
                        
            # 更新监控指标
            self.update_counter.inc(
                value=len(updates),
                labels={"status": "received"}
            )
            
            return updates
            
        except Exception as e:
            logger.error(f"Failed to collect updates: {e}")
            raise
            
    def _aggregate_updates(
        self,
        updates: List[ModelUpdate]
    ) -> Dict[str, np.ndarray]:
        """
        聚合更新
        :param updates: 更新列表
        :return: 全局模型
        """
        start_time = time.time()
        
        try:
            # 解密参数
            decrypted_updates = [
                {
                    name: self._decrypt_param(param)
                    for name, param in update.params.items()
                }
                for update in updates
            ]
            
            if self.strategy == AggregationStrategy.FEDAVG:
                # 联邦平均
                return {
                    name: np.mean(
                        [update[name] for update in decrypted_updates],
                        axis=0
                    )
                    for name in decrypted_updates[0].keys()
                }
                
            elif self.strategy == AggregationStrategy.WEIGHTED:
                # 加权平均
                total_weight = sum(update.weight for update in updates)
                return {
                    name: np.sum(
                        [
                            update.weight * params[name]
                            for update, params in zip(
                                updates,
                                decrypted_updates
                            )
                        ],
                        axis=0
                    ) / total_weight
                    for name in decrypted_updates[0].keys()
                }
                
            elif self.strategy == AggregationStrategy.MEDIAN:
                # 中位数聚合
                return {
                    name: np.median(
                        [update[name] for update in decrypted_updates],
                        axis=0
                    )
                    for name in decrypted_updates[0].keys()
                }
                
            elif self.strategy == AggregationStrategy.TRIMMED:
                # 修剪平均
                trim_ratio = 0.1
                n = len(decrypted_updates)
                k = int(n * trim_ratio)
                return {
                    name: np.mean(
                        sorted(
                            [update[name] for update in decrypted_updates],
                            key=lambda x: np.linalg.norm(x)
                        )[k:-k],
                        axis=0
                    )
                    for name in decrypted_updates[0].keys()
                }
                
            elif self.strategy == AggregationStrategy.KRUM:
                # Krum聚合
                n = len(decrypted_updates)
                m = n - 2  # 假设最多有2个恶意节点
                
                distances = np.zeros((n, n))
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = sum(
                            np.linalg.norm(
                                decrypted_updates[i][name] -
                                decrypted_updates[j][name]
                            )
                            for name in decrypted_updates[0].keys()
                        )
                        distances[i, j] = distances[j, i] = dist
                        
                scores = np.sum(
                    np.partition(distances, m, axis=1)[:, :m],
                    axis=1
                )
                best_idx = np.argmin(scores)
                return decrypted_updates[best_idx]
                
        except Exception as e:
            logger.error(f"Failed to aggregate updates: {e}")
            raise
            
        finally:
            # 更新监控指标
            duration = time.time() - start_time
            self.timing_histogram.observe(
                duration,
                labels={"phase": "aggregation"}
            )
            
    def _update_model(self, global_params: Dict[str, np.ndarray]) -> None:
        """
        更新模型
        :param global_params: 全局参数
        """
        try:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.copy_(torch.from_numpy(global_params[name]))
                    
        except Exception as e:
            logger.error(f"Failed to update model: {e}")
            raise
            
    async def _evaluate_model(self) -> Dict[str, float]:
        """
        评估模型
        :return: 评估指标
        """
        if not self.val_data:
            return {}
            
        start_time = time.time()
        
        try:
            # 设置模型为评估模式
            self.model.eval()
            device = torch.device(self.config.device)
            
            metrics = {}
            total_samples = 0
            
            with torch.no_grad():
                for batch in self.val_data:
                    # 移动数据到设备
                    inputs, targets = self._prepare_batch(batch, device)
                    
                    # 前向传播
                    outputs = self.model(inputs)
                    
                    # 计算指标
                    batch_size = targets.size(0)
                    total_samples += batch_size
                    
                    for metric in self.config.metrics:
                        value = self._compute_metric(
                            metric,
                            outputs,
                            targets
                        )
                        metrics[metric] = metrics.get(metric, 0) + value * batch_size
                        
            # 计算平均指标
            metrics = {
                name: value / total_samples
                for name, value in metrics.items()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise
            
        finally:
            # 更新监控指标
            duration = time.time() - start_time
            self.timing_histogram.observe(
                duration,
                labels={"phase": "evaluation"}
            )
            
    async def _broadcast_global_model(
        self,
        global_params: Dict[str, np.ndarray]
    ) -> None:
        """
        广播全局模型
        :param global_params: 全局参数
        """
        try:
            # 加密参数
            encrypted_params = self._encrypt_params(global_params)
            
            # 广播模型
            await self.network.broadcast(
                "global_model",
                {
                    "round_id": self.current_round,
                    "params": encrypted_params,
                    "timestamp": time.time()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to broadcast global model: {e}")
            raise
            
    async def _handle_model_update(
        self,
        peer_id: str,
        message: Dict
    ) -> None:
        """
        处理模型更新
        :param peer_id: 对端ID
        :param message: 消息
        """
        try:
            # 创建更新
            update = ModelUpdate(
                node_id=peer_id,
                round_id=message["round_id"],
                params=message["params"],
                metrics=message.get("metrics", {}),
                weight=message.get("weight", 1.0),
                timestamp=message["timestamp"]
            )
            
            # 存储更新
            self._pending_updates[peer_id] = update
            
            # 更新监控指标
            self.update_counter.inc(
                labels={"status": "pending"}
            )
            
        except Exception as e:
            logger.error(f"Failed to handle model update: {e}")
            raise
            
    async def _handle_global_model(
        self,
        peer_id: str,
        message: Dict
    ) -> None:
        """
        处理全局模型
        :param peer_id: 对端ID
        :param message: 消息
        """
        try:
            if message["round_id"] != self.current_round:
                return
                
            # 解密参数
            params = {
                name: self._decrypt_param(param)
                for name, param in message["params"].items()
            }
            
            # 更新模型
            self._update_model(params)
            
        except Exception as e:
            logger.error(f"Failed to handle global model: {e}")
            raise
            
    def _encrypt_params(
        self,
        params: Dict[str, np.ndarray]
    ) -> Dict[str, bytes]:
        """
        加密参数
        :param params: 模型参数
        :return: 加密参数
        """
        return {
            name: self.security.encrypt(
                param.tobytes(),
                self.key_pair,
                self.encryption_algorithm
            ).ciphertext
            for name, param in params.items()
        }
        
    def _decrypt_param(
        self,
        encrypted_param: bytes
    ) -> np.ndarray:
        """
        解密参数
        :param encrypted_param: 加密参数
        :return: 解密参数
        """
        decrypted = self.security.decrypt(
            encrypted_param,
            self.key_pair
        )
        return np.frombuffer(decrypted, dtype=np.float32)
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        创建优化器
        :return: 优化器
        """
        if self.config.optimizer.lower() == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
        elif self.config.optimizer.lower() == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
            
    def _create_loss_function(self) -> nn.Module:
        """
        创建损失函数
        :return: 损失函数
        """
        if self.config.loss_fn.lower() == "mse":
            return nn.MSELoss()
        elif self.config.loss_fn.lower() == "cross_entropy":
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.config.loss_fn}")
            
    def _prepare_batch(
        self,
        batch: Tuple,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        准备批次数据
        :param batch: 数据批次
        :param device: 设备
        :return: 输入和目标
        """
        if len(batch) != 2:
            raise ValueError("Batch must contain inputs and targets")
            
        inputs, targets = batch
        return inputs.to(device), targets.to(device)
        
    def _compute_metric(
        self,
        metric: str,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """
        计算指标
        :param metric: 指标名称
        :param outputs: 模型输出
        :param targets: 目标值
        :return: 指标值
        """
        if metric.lower() == "accuracy":
            predictions = outputs.argmax(dim=1)
            return (predictions == targets).float().mean().item()
        elif metric.lower() == "mse":
            return nn.MSELoss()(outputs, targets).item()
        else:
            raise ValueError(f"Unsupported metric: {metric}")
            
    def _update_best_metrics(self, metrics: Dict[str, float]) -> None:
        """
        更新最佳指标
        :param metrics: 评估指标
        """
        for name, value in metrics.items():
            if (name not in self.best_metrics or
                value > self.best_metrics[name]):
                self.best_metrics[name] = value

def create_federated_learner(
    network: P2PNetwork,
    security: SecurityManager,
    model: nn.Module,
    train_data: DataLoader,
    **kwargs
) -> FederatedLearner:
    """
    创建联邦学习器
    :param network: P2P网络
    :param security: 安全管理器
    :param model: 模型
    :param train_data: 训练数据
    :param kwargs: 附加参数
    :return: 联邦学习器实例
    """
    return FederatedLearner(
        network=network,
        security=security,
        model=model,
        train_data=train_data,
        **kwargs
    )
