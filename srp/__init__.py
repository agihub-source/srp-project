"""
丝路协议(Silk Road Protocol, SRP)
一个创新的去中心化AI系统通信协议
"""

__version__ = "0.1.0"
__author__ = "SRP Contributors"
__license__ = "MIT"

from .sdk.client import SRPClient
from .network.p2p import SRPNode
from .security.encryption import Encryptor, HomomorphicEncryption, SecureDataManager
from .compliance.compliance import (
    ComplianceChecker,
    DataComplianceValidator,
    check_data_compliance,
    DataLocationType,
    DataSensitivityLevel,
    AIEthicsLevel
)
from .state.session import StateManager
from .network.adapters import ProtobufAdapter, JsonRpcAdapter
from .plugins.plugin_manager import SRPPlugin

# 版本信息
VERSION = __version__

# 导出主要类和函数
__all__ = [
    'SRPClient',           # 客户端SDK
    'SRPNode',            # P2P节点
    'Encryptor',          # 加密器
    'HomomorphicEncryption',  # 同态加密
    'SecureDataManager',  # 安全数据管理
    'ComplianceChecker',  # 合规性检查器
    'DataComplianceValidator',  # 数据合规性验证
    'check_data_compliance',   # 合规性检查函数
    'DataLocationType',    # 数据位置类型
    'DataSensitivityLevel',  # 数据敏感度级别
    'AIEthicsLevel',      # AI伦理级别
    'StateManager',       # 状态管理器
    'ProtobufAdapter',    # Protobuf适配器
    'JsonRpcAdapter',     # JSON-RPC适配器
    'SRPPlugin',         # 插件基类
]

# 默认配置
DEFAULT_CONFIG = {
    'host': '127.0.0.1',
    'port': 8000,
    'bootstrap_peers': [],
    'encryption_enabled': True,
    'compliance_check': True,
    'data_location': 'China',
    'log_level': 'INFO'
}

def get_version():
    """获取版本信息"""
    return __version__

def get_config():
    """获取默认配置"""
    return DEFAULT_CONFIG.copy()