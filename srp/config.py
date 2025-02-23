"""配置管理模块"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .monitoring import monitoring

logger = logging.getLogger(__name__)

class NetworkMode(Enum):
    """网络模式"""
    P2P = auto()      # 点对点模式
    RELAY = auto()    # 中继模式
    HYBRID = auto()   # 混合模式

class SecurityLevel(Enum):
    """安全级别"""
    LOW = auto()      # 低安全级别
    MEDIUM = auto()   # 中等安全级别
    HIGH = auto()     # 高安全级别

@dataclass
class NetworkConfig:
    """网络配置"""
    mode: NetworkMode = NetworkMode.P2P         # 网络模式
    host: str = "127.0.0.1"                     # 监听主机
    port: int = 8000                            # 监听端口
    max_connections: int = 100                  # 最大连接数
    timeout: int = 30                           # 超时时间(秒)
    relay_servers: List[str] = field(          # 中继服务器列表
        default_factory=lambda: []
    )
    bootstrap_nodes: List[str] = field(        # 引导节点列表
        default_factory=lambda: []
    )

@dataclass
class SecurityConfig:
    """安全配置"""
    level: SecurityLevel = SecurityLevel.MEDIUM  # 安全级别
    enable_encryption: bool = True              # 启用加密
    key_rotation: int = 3600                    # 密钥轮换周期(秒)
    tls_enabled: bool = True                    # 启用TLS
    tls_cert_file: str = "cert.pem"            # TLS证书文件
    tls_key_file: str = "key.pem"              # TLS密钥文件
    allowed_ciphers: List[str] = field(        # 允许的加密算法
        default_factory=lambda: [
            "aes-gcm",
            "aes-cbc",
            "ec"
        ]
    )

@dataclass
class StorageConfig:
    """存储配置"""
    path: str = "./data"                        # 数据存储路径
    max_size: int = 1024 * 1024 * 1024         # 最大存储空间(字节)
    cleanup_interval: int = 3600                # 清理间隔(秒)
    compression: bool = True                    # 启用压缩
    encryption: bool = True                     # 启用加密

@dataclass
class ComplianceConfig:
    """合规配置"""
    region: str = "CN"                          # 地区
    data_retention_days: int = 180             # 数据保留天数
    enable_audit: bool = True                  # 启用审计
    audit_log_path: str = "./audit.log"        # 审计日志路径
    restricted_ips: List[str] = field(         # 限制IP列表
        default_factory=lambda: []
    )

@dataclass
class Config:
    """系统配置"""
    network: NetworkConfig = field(
        default_factory=NetworkConfig
    )
    security: SecurityConfig = field(
        default_factory=SecurityConfig
    )
    storage: StorageConfig = field(
        default_factory=StorageConfig
    )
    compliance: ComplianceConfig = field(
        default_factory=ComplianceConfig
    )

class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self._config = Config()
        self._config_file: Optional[str] = None
        self._env_prefix = "SRP_"
        
        # 监控指标
        self.reload_counter = monitoring.counter(
            "config_reloads_total",
            "Total number of configuration reloads",
            ["status"]
        )
        
    @property
    def config(self) -> Config:
        """获取配置"""
        return self._config
        
    def load_file(self, file_path: Union[str, Path]) -> None:
        """
        从文件加载配置
        :param file_path: 配置文件路径
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                
            self._update_config(config_data)
            self._config_file = str(file_path)
            
            self.reload_counter.inc(
                labels={"status": "success"}
            )
            logger.info(f"Loaded config from: {file_path}")
            
        except Exception as e:
            self.reload_counter.inc(
                labels={"status": "error"}
            )
            logger.error(f"Failed to load config: {e}")
            raise
            
    def reload(self) -> None:
        """重新加载配置"""
        try:
            if not self._config_file:
                return
                
            self.load_file(self._config_file)
            self._load_env_vars()
            
            self.reload_counter.inc(
                labels={"status": "success"}
            )
            logger.info("Reloaded config")
            
        except Exception as e:
            self.reload_counter.inc(
                labels={"status": "error"}
            )
            logger.error(f"Failed to reload config: {e}")
            raise
            
    def save_file(self, file_path: Union[str, Path]) -> None:
        """
        保存配置到文件
        :param file_path: 配置文件路径
        """
        try:
            config_data = self._serialize_config()
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2)
                
            self._config_file = str(file_path)
            logger.info(f"Saved config to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise
            
    def _update_config(self, data: Dict[str, Any]) -> None:
        """
        更新配置
        :param data: 配置数据
        """
        # 网络配置
        if "network" in data:
            net_conf = data["network"]
            if "mode" in net_conf:
                self._config.network.mode = NetworkMode[net_conf["mode"]]
            if "host" in net_conf:
                self._config.network.host = net_conf["host"]
            if "port" in net_conf:
                self._config.network.port = int(net_conf["port"])
            if "max_connections" in net_conf:
                self._config.network.max_connections = int(net_conf["max_connections"])
            if "timeout" in net_conf:
                self._config.network.timeout = int(net_conf["timeout"])
            if "relay_servers" in net_conf:
                self._config.network.relay_servers = list(net_conf["relay_servers"])
            if "bootstrap_nodes" in net_conf:
                self._config.network.bootstrap_nodes = list(net_conf["bootstrap_nodes"])
                
        # 安全配置
        if "security" in data:
            sec_conf = data["security"]
            if "level" in sec_conf:
                self._config.security.level = SecurityLevel[sec_conf["level"]]
            if "enable_encryption" in sec_conf:
                self._config.security.enable_encryption = bool(sec_conf["enable_encryption"])
            if "key_rotation" in sec_conf:
                self._config.security.key_rotation = int(sec_conf["key_rotation"])
            if "tls_enabled" in sec_conf:
                self._config.security.tls_enabled = bool(sec_conf["tls_enabled"])
            if "tls_cert_file" in sec_conf:
                self._config.security.tls_cert_file = sec_conf["tls_cert_file"]
            if "tls_key_file" in sec_conf:
                self._config.security.tls_key_file = sec_conf["tls_key_file"]
            if "allowed_ciphers" in sec_conf:
                self._config.security.allowed_ciphers = list(sec_conf["allowed_ciphers"])
                
        # 存储配置
        if "storage" in data:
            store_conf = data["storage"]
            if "path" in store_conf:
                self._config.storage.path = store_conf["path"]
            if "max_size" in store_conf:
                self._config.storage.max_size = int(store_conf["max_size"])
            if "cleanup_interval" in store_conf:
                self._config.storage.cleanup_interval = int(store_conf["cleanup_interval"])
            if "compression" in store_conf:
                self._config.storage.compression = bool(store_conf["compression"])
            if "encryption" in store_conf:
                self._config.storage.encryption = bool(store_conf["encryption"])
                
        # 合规配置
        if "compliance" in data:
            comp_conf = data["compliance"]
            if "region" in comp_conf:
                self._config.compliance.region = comp_conf["region"]
            if "data_retention_days" in comp_conf:
                self._config.compliance.data_retention_days = int(comp_conf["data_retention_days"])
            if "enable_audit" in comp_conf:
                self._config.compliance.enable_audit = bool(comp_conf["enable_audit"])
            if "audit_log_path" in comp_conf:
                self._config.compliance.audit_log_path = comp_conf["audit_log_path"]
            if "restricted_ips" in comp_conf:
                self._config.compliance.restricted_ips = list(comp_conf["restricted_ips"])
                
    def _load_env_vars(self) -> None:
        """从环境变量加载配置"""
        try:
            # 网络配置
            if host := os.getenv(f"{self._env_prefix}HOST"):
                self._config.network.host = host
            if port := os.getenv(f"{self._env_prefix}PORT"):
                self._config.network.port = int(port)
                
            # 安全配置
            if level := os.getenv(f"{self._env_prefix}SECURITY_LEVEL"):
                self._config.security.level = SecurityLevel[level]
            if cert_file := os.getenv(f"{self._env_prefix}TLS_CERT_FILE"):
                self._config.security.tls_cert_file = cert_file
            if key_file := os.getenv(f"{self._env_prefix}TLS_KEY_FILE"):
                self._config.security.tls_key_file = key_file
                
            # 存储配置
            if path := os.getenv(f"{self._env_prefix}STORAGE_PATH"):
                self._config.storage.path = path
                
            # 合规配置
            if region := os.getenv(f"{self._env_prefix}REGION"):
                self._config.compliance.region = region
                
            logger.info("Loaded config from environment variables")
            
        except Exception as e:
            logger.error(f"Failed to load config from environment: {e}")
            raise
            
    def _serialize_config(self) -> Dict[str, Any]:
        """
        序列化配置
        :return: 配置数据
        """
        config_dict = asdict(self._config)
        
        # 处理枚举值
        if "network" in config_dict:
            if "mode" in config_dict["network"]:
                config_dict["network"]["mode"] = config_dict["network"]["mode"].name
                
        if "security" in config_dict:
            if "level" in config_dict["security"]:
                config_dict["security"]["level"] = config_dict["security"]["level"].name
                
        return config_dict

# 创建全局配置管理器实例
config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """
    获取配置管理器实例
    :return: 配置管理器实例
    """
    global config_manager
    if not config_manager:
        config_manager = ConfigManager()
    return config_manager

# 初始化全局配置管理器
config_manager = get_config_manager()
