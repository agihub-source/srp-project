# SRP API文档

## 核心模块

### 1. SDK (srp.sdk.client)

#### SRPClient

主要客户端类，用于管理P2P网络连接和消息通信。

```python
class SRPClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        config: Optional[Dict] = None
    ):
        """
        初始化SRP客户端
        
        参数:
            host: 监听地址
            port: 监听端口
            config: 配置字典
        """
        pass

    async def start(self) -> None:
        """启动客户端"""
        pass

    async def stop(self) -> None:
        """停止客户端"""
        pass

    def create_session(self, metadata: Dict) -> str:
        """
        创建新会话
        
        参数:
            metadata: 会话元数据
            
        返回:
            session_id: 会话ID
        """
        pass

    async def send_message(
        self,
        peer_id: str,
        message_type: str,
        data: Dict,
        timeout: float = 30.0
    ) -> Dict:
        """
        发送消息到指定节点
        
        参数:
            peer_id: 目标节点ID
            message_type: 消息类型
            data: 消息数据
            timeout: 超时时间(秒)
            
        返回:
            响应数据
        """
        pass
```

### 2. 网络层 (srp.network)

#### RoutingManager

负责P2P网络中的路由管理。

```python
class RoutingManager:
    def __init__(
        self,
        node_id: str,
        secret_key: bytes,
        max_bandwidth: float = 1e6
    ):
        """
        初始化路由管理器
        
        参数:
            node_id: 节点ID
            secret_key: 密钥
            max_bandwidth: 最大带宽限制
        """
        pass

    async def find_path(
        self,
        target_id: str,
        min_success_rate: float = 0.5,
        max_latency: float = 1.0
    ) -> Optional[List[str]]:
        """
        查找到目标节点的最优路径
        
        参数:
            target_id: 目标节点ID
            min_success_rate: 最低成功率要求
            max_latency: 最大延迟要求
            
        返回:
            节点ID列表表示的路径
        """
        pass
```

### 3. 安全层 (srp.security)

#### Encryptor

提供加密和解密功能。

```python
class Encryptor:
    def __init__(self, secret_key: bytes):
        """
        初始化加密器
        
        参数:
            secret_key: 密钥
        """
        pass

    def encrypt(self, data: bytes) -> bytes:
        """
        加密数据
        
        参数:
            data: 待加密数据
            
        返回:
            加密后的数据
        """
        pass

    def decrypt(self, data: bytes) -> bytes:
        """
        解密数据
        
        参数:
            data: 待解密数据
            
        返回:
            解密后的数据
        """
        pass
```

### 4. 联邦学习 (srp.federated)

#### FederatedLearner

负责联邦学习的训练和模型聚合。

```python
class FederatedLearner:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Type[optim.Optimizer],
        loss_fn: nn.Module
    ):
        """
        初始化联邦学习器
        
        参数:
            model: PyTorch模型
            optimizer: 优化器类
            loss_fn: 损失函数
        """
        pass

    async def train_round(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        执行一轮本地训练
        
        参数:
            data: 训练数据
            labels: 标签
            epochs: 训练轮数
            
        返回:
            更新的模型参数
        """
        pass

    async def aggregate_updates(
        self,
        updates: List[Dict[str, torch.Tensor]]
    ) -> None:
        """
        聚合来自其他节点的模型更新
        
        参数:
            updates: 模型参数更新列表
        """
        pass
```

### 5. 插件系统 (srp.plugins)

#### PluginManager

管理插件的生命周期。

```python
class PluginManager:
    def __init__(self):
        """初始化插件管理器"""
        pass

    def load_plugin(self, name: str) -> bool:
        """
        加载插件
        
        参数:
            name: 插件名称
            
        返回:
            是否加载成功
        """
        pass

    def unload_plugin(self, name: str) -> bool:
        """
        卸载插件
        
        参数:
            name: 插件名称
            
        返回:
            是否卸载成功
        """
        pass
```

## 错误处理

所有模块都可能抛出以下异常：

```python
class SRPError(Exception):
    """SRP基础异常类"""
    pass

class NetworkError(SRPError):
    """网络相关错误"""
    pass

class SecurityError(SRPError):
    """安全相关错误"""
    pass

class ValidationError(SRPError):
    """数据验证错误"""
    pass
```

## 配置选项

配置通过字典传递，支持以下选项：

```python
config = {
    "network": {
        "max_peers": 50,        # 最大连接节点数
        "bootstrap_nodes": [],  # 引导节点列表
        "timeout": 30.0,       # 默认超时时间
    },
    "security": {
        "encryption_algo": "AES-256-GCM",  # 加密算法
        "key_rotation": 3600,              # 密钥轮换间隔(秒)
    },
    "federated": {
        "min_peers": 3,        # 最小参与节点数
        "round_timeout": 600,  # 训练轮次超时时间
    }
}
```

## 使用示例

参见 [examples/](../examples/) 目录下的示例代码。

## 性能考虑

1. 网络性能
   - 消息大小建议控制在1MB以内
   - 单节点并发连接数建议不超过100
   - 路由表更新间隔建议设置为1-5秒

2. 加密性能
   - 密钥轮换不要过于频繁
   - 大量数据传输时考虑使用流式加密
   - 合理使用缓存减少重复加密

3. 联邦学习
   - 本地训练批次大小建议为32-128
   - 模型更新压缩可减少网络负载
   - 合理设置训练轮次避免过度通信