"""
SRP客户端SDK - 提供简单易用的API接口
包含完整的安全机制和隐私保护
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import asyncio
import os
import time
import hmac
import hashlib
import logging
import torch
import torch.nn as nn
from torch.optim import Optimizer
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..network.p2p import SRPNode
from ..security.encryption import Encryptor, HomomorphicEncryption
from ..compliance.compliance import check_data_compliance
from ..state.session import StateManager
from ..network.routing import NodeInfo

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClientError(Exception):
    """客户端错误基类"""
    pass

class SecurityError(ClientError):
    """安全相关错误"""
    pass

class AuthenticationError(SecurityError):
    """认证错误"""
    pass

class DifferentialPrivacy:
    """差分隐私实现"""
    
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
    
    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """添加拉普拉斯噪声"""
        sensitivity = 1.0
        scale = sensitivity / self.epsilon
        noise = torch.distributions.Laplace(0, scale).sample(tensor.shape)
        return tensor + noise

class SecureOptimizer:
    """安全优化器"""
    
    def __init__(
        self,
        optimizer: Optimizer,
        dp: DifferentialPrivacy,
        encryption: HomomorphicEncryption
    ):
        self.optimizer = optimizer
        self.dp = dp
        self.encryption = encryption
    
    def step(self):
        """执行安全的梯度更新"""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # 添加差分隐私噪声
                noised_grad = self.dp.add_noise(p.grad.data)
                
                # 加密梯度
                encrypted_grad = self.encryption.encrypt(
                    noised_grad.numpy().tolist()
                )
                
                # 更新参数
                p.data.add_(encrypted_grad, alpha=-group['lr'])

class SRPClient:
    """SRP客户端实现"""
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        identity_file: Optional[str] = None
    ):
        """
        初始化SRP客户端
        :param host: 主机地址
        :param port: 端口号
        :param identity_file: 身份文件路径
        """
        self.node = SRPNode(host=host, port=port)
        self.encryptor = Encryptor()
        self.homomorphic = HomomorphicEncryption()
        self.state_manager = StateManager()
        self.dp = DifferentialPrivacy()
        
        self._federated_model = None
        self._worker_id = None
        self._running = False
        self._session_token = None
        self._session_expiry = 0
        self._auth_key = None
        
        # 加载身份信息
        self._load_identity(identity_file)
        
        # 创建线程池
        self.executor = ThreadPoolExecutor(max_workers=5)

    def _load_identity(self, identity_file: Optional[str]):
        """加载身份信息"""
        try:
            if identity_file:
                path = Path(identity_file).resolve()
                if not path.exists():
                    raise SecurityError("身份文件不存在")
                
                with open(path, 'rb') as f:
                    self._auth_key = f.read()
            else:
                self._auth_key = os.urandom(32)
                
        except Exception as e:
            logger.error("加载身份信息失败")
            raise SecurityError(str(e))

    async def authenticate(self) -> bool:
        """身份认证"""
        try:
            # 生成随机数
            nonce = os.urandom(16)
            timestamp = int(time.time())
            
            # 创建签名
            message = b''.join([
                self._auth_key,
                nonce,
                str(timestamp).encode()
            ])
            signature = hmac.new(
                self._auth_key,
                message,
                hashlib.sha256
            ).digest()
            
            # 发送认证请求
            response = await self.node.send_message(
                "authentication",
                {
                    "nonce": nonce.hex(),
                    "timestamp": timestamp,
                    "signature": signature.hex()
                }
            )
            
            if not response or "session_token" not in response:
                raise AuthenticationError("认证失败")
            
            # 保存会话信息
            self._session_token = response["session_token"]
            self._session_expiry = response["expiry"]
            
            return True
            
        except Exception as e:
            logger.error("认证失败")
            raise AuthenticationError(str(e))

    async def start(self):
        """启动客户端"""
        if self._running:
            return
        
        try:
            # 启动网络节点
            await self.node.start()
            
            # 进行身份认证
            if not await self.authenticate():
                raise SecurityError("身份认证失败")
            
            # 启动状态管理器
            await self.state_manager.start_cleanup()
            
            self._running = True
            logger.info(f"SRP客户端启动于 {self.node.host}:{self.node.port}")
            
        except Exception as e:
            logger.error("客户端启动失败")
            raise ClientError(str(e))

    async def stop(self):
        """停止客户端"""
        if not self._running:
            return
            
        try:
            await self.node.stop()
            await self.state_manager.stop_cleanup()
            self.executor.shutdown()
            
            self._running = False
            self._session_token = None
            logger.info("SRP客户端已停止")
            
        except Exception as e:
            logger.error("客户端停止失败")
            raise ClientError(str(e))

    def _check_session(self):
        """检查会话是否有效"""
        if not self._session_token or time.time() > self._session_expiry:
            raise AuthenticationError("会话已过期")

    async def connect_to(self, peer_address: str) -> bool:
        """
        连接到对等节点
        :param peer_address: 对等节点地址
        :return: 是否连接成功
        """
        try:
            self._check_session()
            return await self.node.connect_peer(peer_address)
        except Exception as e:
            logger.error(f"连接节点失败: {peer_address}")
            raise ClientError(str(e))

    async def send_message(
        self,
        peer_id: str,
        message_type: str,
        data: Dict[str, Any],
        use_json_rpc: bool = False,
        encrypt: bool = True  # 默认加密
    ) -> Optional[Dict[str, Any]]:
        """
        发送消息到指定节点
        :param peer_id: 目标节点ID
        :param message_type: 消息类型
        :param data: 消息数据
        :param use_json_rpc: 是否使用JSON-RPC
        :param encrypt: 是否加密数据
        :return: 响应数据
        """
        try:
            self._check_session()
            
            # 准备消息数据
            message_data = {
                "session_token": self._session_token,
                "timestamp": int(time.time())
            }
            
            if encrypt:
                # 生成消息ID和HMAC
                message_id = os.urandom(16).hex()
                message_hmac = hmac.new(
                    self._auth_key,
                    str(data).encode(),
                    hashlib.sha256
                ).hexdigest()
                
                # 加密数据
                encrypted_data = self.encryptor.encrypt(str(data))
                
                message_data.update({
                    "message_id": message_id,
                    "hmac": message_hmac,
                    "encrypted_data": encrypted_data
                })
            else:
                message_data.update(data)
            
            return await self.node.send_message(
                peer_id,
                message_type,
                message_data,
                use_json_rpc
            )
            
        except Exception as e:
            logger.error("消息发送失败")
            raise ClientError(str(e))

    async def init_federated_learning(
        self,
        model: nn.Module,
        worker_id: str,
        dp_epsilon: float = 1.0
    ):
        """
        初始化联邦学习
        :param model: PyTorch模型
        :param worker_id: 工作节点ID
        :param dp_epsilon: 差分隐私参数
        """
        try:
            self._check_session()
            
            self._federated_model = model
            self._worker_id = worker_id
            
            # 更新差分隐私参数
            self.dp = DifferentialPrivacy(dp_epsilon)
            
            logger.info(f"初始化联邦学习 - 工作节点: {worker_id}")
            
        except Exception as e:
            logger.error("联邦学习初始化失败")
            raise ClientError(str(e))

    async def participate_training(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 0.01
    ):
        """
        参与联邦学习训练
        :param data: 训练数据
        :param target: 目标数据
        :param epochs: 训练轮数
        :param batch_size: 批次大小
        :param learning_rate: 学习率
        """
        try:
            self._check_session()
            
            if not self._federated_model or not self._worker_id:
                raise RuntimeError("请先调用init_federated_learning")
            
            # 创建安全优化器
            optimizer = torch.optim.SGD(
                self._federated_model.parameters(),
                lr=learning_rate
            )
            secure_optimizer = SecureOptimizer(
                optimizer,
                self.dp,
                self.homomorphic
            )
            
            # 训练循环
            for epoch in range(epochs):
                epoch_loss = 0.0
                for i in range(0, len(data), batch_size):
                    # 准备批次数据
                    batch_data = data[i:i + batch_size]
                    batch_target = target[i:i + batch_size]
                    
                    # 前向传播
                    optimizer.zero_grad()
                    output = self._federated_model(batch_data)
                    loss = torch.nn.functional.cross_entropy(
                        output,
                        batch_target
                    )
                    
                    # 反向传播
                    loss.backward()
                    
                    # 安全更新
                    secure_optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    if i % 100 == 0:
                        logger.info(
                            f"Epoch {epoch+1}, "
                            f"Step {i}, "
                            f"Loss: {loss.item():.4f}"
                        )
                
                # 加密并上传模型更新
                await self._upload_model_update(epoch, epoch_loss / len(data))
                
        except Exception as e:
            logger.error("联邦学习训练失败")
            raise ClientError(str(e))

    async def _upload_model_update(
        self,
        epoch: int,
        loss: float
    ):
        """
        上传模型更新
        :param epoch: 当前轮次
        :param loss: 损失值
        """
        try:
            # 收集模型参数
            params = []
            for param in self._federated_model.parameters():
                params.append(param.data.numpy().tolist())
            
            # 加密模型参数
            encrypted_params = self.homomorphic.encrypt(str(params))
            
            # 创建更新消息
            update = {
                "worker_id": self._worker_id,
                "epoch": epoch,
                "loss": loss,
                "encrypted_params": encrypted_params
            }
            
            # 发送更新
            await self.send_message(
                "aggregator",
                "model_update",
                update
            )
            
        except Exception as e:
            logger.error("模型更新上传失败")
            raise ClientError(str(e))

    def create_session(self, initial_data: Dict[str, Any] = None) -> str:
        """
        创建会话
        :param initial_data: 初始状态数据
        :return: 会话ID
        """
        try:
            self._check_session()
            
            # 加密初始数据
            if initial_data:
                initial_data = {
                    k: self.encryptor.encrypt(str(v))
                    for k, v in initial_data.items()
                }
            
            return self.state_manager.create_session(initial_data)
            
        except Exception as e:
            logger.error("会话创建失败")
            raise ClientError(str(e))

    def check_compliance(
        self,
        data: Dict[str, Any],
        strict: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        检查数据合规性
        :param data: 要检查的数据
        :param strict: 是否严格模式
        :return: (是否合规, 违规原因列表)
        """
        try:
            issues = []
            compliant = True
            
            try:
                check_data_compliance(data)
            except ValueError as e:
                issues.append(str(e))
                if strict:
                    compliant = False
            
            return compliant, issues
            
        except Exception as e:
            logger.error("合规性检查失败")
            raise ClientError(str(e))

    # 其他方法类似地添加安全检查...