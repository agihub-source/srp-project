"""
SRP联邦学习示例
展示了如何使用SRP进行隐私保护的分布式机器学习

功能特点：
- 本地模型训练
- 安全模型聚合
- 梯度加密
- 差分隐私
"""

import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List
import logging

from srp.sdk.client import SRPClient
from srp.federated.federated_learning import FederatedLearner
from srp.security.encryption import Encryptor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleModel(nn.Module):
    """
    简单的神经网络模型
    """
    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

class PrivacyGuard:
    """
    隐私保护机制
    """
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
        
    def add_noise(self, gradients: torch.Tensor) -> torch.Tensor:
        """添加差分隐私噪声"""
        sensitivity = 1.0
        noise_scale = sensitivity / self.epsilon
        noise = torch.normal(
            mean=0.0,
            std=noise_scale,
            size=gradients.shape,
            device=gradients.device
        )
        return gradients + noise

class FederatedClient:
    """
    联邦学习客户端
    """
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        privacy_budget: float = 1.0
    ):
        self.client_id = client_id
        self.model = model
        self.train_data = train_data
        self.train_labels = train_labels
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.privacy_guard = PrivacyGuard(privacy_budget)
        
        # 创建数据加载器
        self.train_dataset = TensorDataset(train_data, train_labels)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    async def train_local(self, epochs: int = 1) -> Dict[str, torch.Tensor]:
        """本地训练"""
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                # 添加差分隐私
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad = self.privacy_guard.add_noise(param.grad)
                        
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                if batch_idx % 10 == 0:
                    logger.info(
                        f"Epoch: {epoch}, Batch: {batch_idx}, "
                        f"Loss: {loss.item():.4f}, "
                        f"Acc: {100.*correct/total:.2f}%"
                    )
                    
        # 收集模型更新
        updates = {
            name: param.data.clone()
            for name, param in self.model.state_dict().items()
        }
        
        return updates

async def run_federated_example():
    """运行联邦学习示例"""
    # 创建模拟数据
    num_clients = 3
    input_size = 784  # 28x28
    num_samples = 1000
    num_classes = 10
    
    # 生成随机数据
    torch.manual_seed(42)
    data_list = []
    label_list = []
    for _ in range(num_clients):
        data = torch.randn(num_samples, input_size)
        labels = torch.randint(0, num_classes, (num_samples,))
        data_list.append(data)
        label_list.append(labels)
    
    # 创建客户端
    clients = []
    models = []
    for i in range(num_clients):
        model = SimpleModel(input_size, 128, num_classes)
        client = FederatedClient(
            client_id=f"client_{i}",
            model=model,
            train_data=data_list[i],
            train_labels=label_list[i]
        )
        clients.append(client)
        models.append(model)
    
    # 创建SRP客户端和联邦学习管理器
    srp_clients = []
    for i in range(num_clients):
        client = SRPClient(
            host="127.0.0.1",
            port=8000+i
        )
        await client.start()
        srp_clients.append(client)
    
    try:
        # 运行联邦学习
        num_rounds = 5
        for round_idx in range(num_rounds):
            logger.info(f"=== 开始训练轮次 {round_idx + 1}/{num_rounds} ===")
            
            # 1. 本地训练
            updates_list = []
            for i, client in enumerate(clients):
                logger.info(f"客户端 {i} 开始本地训练...")
                updates = await client.train_local(epochs=1)
                updates_list.append(updates)
                
            # 2. 加密模型更新
            encrypted_updates = []
            for i, updates in enumerate(updates_list):
                encryptor = Encryptor(srp_clients[i].get_session_key("server"))
                encrypted = {
                    name: encryptor.encrypt(param.numpy().tobytes())
                    for name, param in updates.items()
                }
                encrypted_updates.append(encrypted)
                
            # 3. 发送加密更新
            for i, encrypted in enumerate(encrypted_updates):
                await srp_clients[i].send_message(
                    peer_id="server",
                    message_type="model_update",
                    data={
                        "round": round_idx,
                        "client_id": f"client_{i}",
                        "updates": encrypted
                    }
                )
                
            # 4. 模拟服务器聚合
            # 在实际应用中，这部分由服务器完成
            logger.info("服务器聚合模型更新...")
            
            # 解密更新
            decrypted_updates = []
            for encrypted in encrypted_updates:
                decrypted = {
                    name: torch.from_numpy(
                        np.frombuffer(
                            Encryptor(b"test_key").decrypt(param),
                            dtype=np.float32
                        ).reshape(models[0].state_dict()[name].shape)
                    )
                    for name, param in encrypted.items()
                }
                decrypted_updates.append(decrypted)
                
            # 平均聚合
            aggregated_update = {}
            for name in decrypted_updates[0].keys():
                aggregated_update[name] = torch.stack([
                    update[name] for update in decrypted_updates
                ]).mean(dim=0)
                
            # 5. 更新本地模型
            for client in clients:
                client.model.load_state_dict(aggregated_update)
                
            logger.info(f"=== 完成训练轮次 {round_idx + 1}/{num_rounds} ===\n")
            
    finally:
        # 关闭SRP客户端
        for client in srp_clients:
            await client.stop()

def main():
    """主函数"""
    try:
        asyncio.run(run_federated_example())
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序异常退出: {str(e)}")

if __name__ == "__main__":
    main()