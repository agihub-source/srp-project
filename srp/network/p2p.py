"""P2P网络模块"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .adapters import ConnectionInfo, NetworkAdapter, TCPAdapter
from ..monitoring import monitoring

logger = logging.getLogger(__name__)

class NodeRole(Enum):
    """节点角色"""
    NORMAL = auto()      # 普通节点
    SUPER = auto()       # 超级节点
    BOOTSTRAP = auto()   # 引导节点
    RELAY = auto()       # 中继节点

class NodeState(Enum):
    """节点状态"""
    OFFLINE = auto()    # 离线
    CONNECTING = auto() # 连接中
    ONLINE = auto()     # 在线
    LEAVING = auto()    # 离开中

@dataclass
class NodeInfo:
    """节点信息"""
    node_id: str              # 节点ID
    address: str              # 地址
    port: int                # 端口
    role: NodeRole           # 角色
    state: NodeState         # 状态
    capabilities: Set[str]   # 能力
    metadata: Dict = field(default_factory=dict)  # 元数据

@dataclass
class RoutingTableEntry:
    """路由表项"""
    node: NodeInfo            # 节点信息
    distance: float          # 距离
    latency: float          # 延迟
    last_seen: float        # 最后可见时间
    hops: int = 1           # 跳数
    reliability: float = 1.0 # 可靠性

class P2PNetwork:
    """P2P网络"""
    
    def __init__(
        self,
        node_id: str,
        host: str,
        port: int,
        role: NodeRole = NodeRole.NORMAL,
        bootstrap_nodes: List[Tuple[str, int]] = None,
        max_connections: int = 50,
        refresh_interval: int = 60
    ):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.role = role
        self.bootstrap_nodes = bootstrap_nodes or []
        self.max_connections = max_connections
        self.refresh_interval = refresh_interval
        
        # 创建TCP适配器
        self.adapter = TCPAdapter(host, port)
        
        # 节点状态
        self.state = NodeState.OFFLINE
        self.capabilities: Set[str] = set()
        self.metadata: Dict = {}
        
        # 路由表
        self._routing_table: Dict[str, RoutingTableEntry] = {}
        self._pending_requests: Dict[str, asyncio.Future] = {}
        
        # 消息处理器
        self._message_handlers: Dict[str, List[Callable]] = {}
        
        # 监控指标
        self.node_gauge = monitoring.gauge(
            "srp_p2p_nodes",
            "Number of P2P nodes",
            ["role", "state"]
        )
        
        self.routing_gauge = monitoring.gauge(
            "srp_routing_entries",
            "Number of routing table entries",
            ["role"]
        )
        
        self.latency_histogram = monitoring.histogram(
            "srp_p2p_latency_seconds",
            "P2P network latency in seconds",
            ["node_id"]
        )
        
        self.reliability_histogram = monitoring.histogram(
            "srp_p2p_reliability",
            "P2P network reliability score",
            ["node_id"]
        )
        
        self.message_counter = monitoring.counter(
            "srp_p2p_messages_total",
            "Total number of P2P messages",
            ["type", "direction"]
        )
        
    async def start(self) -> None:
        """启动P2P网络"""
        try:
            # 启动网络适配器
            await self.adapter.start()
            
            # 注册消息处理器
            self.adapter.register_event_handler(
                "client_connected",
                self._handle_client_connected
            )
            self.adapter.register_event_handler(
                "client_disconnected",
                self._handle_client_disconnected
            )
            self.adapter.register_event_handler(
                "data_received",
                self._handle_data_received
            )
            
            # 更新状态
            self.state = NodeState.CONNECTING
            
            # 连接引导节点
            for address, port in self.bootstrap_nodes:
                await self._connect_bootstrap_node(address, port)
                
            # 启动定期刷新
            asyncio.create_task(self._periodic_refresh())
            
            # 更新状态
            self.state = NodeState.ONLINE
            
            # 更新监控指标
            self.node_gauge.inc(
                labels={
                    "role": self.role.name.lower(),
                    "state": self.state.name.lower()
                }
            )
            
            logger.info(f"P2P network started: {self.node_id}")
            
        except Exception as e:
            logger.error(f"Failed to start P2P network: {e}")
            raise
            
    async def stop(self) -> None:
        """停止P2P网络"""
        try:
            # 更新状态
            self.state = NodeState.LEAVING
            
            # 通知邻居节点
            await self._broadcast_leave()
            
            # 关闭所有连接
            await self.adapter.stop()
            
            # 更新状态
            self.state = NodeState.OFFLINE
            
            # 更新监控指标
            self.node_gauge.dec(
                labels={
                    "role": self.role.name.lower(),
                    "state": NodeState.ONLINE.name.lower()
                }
            )
            
            logger.info(f"P2P network stopped: {self.node_id}")
            
        except Exception as e:
            logger.error(f"Failed to stop P2P network: {e}")
            raise
            
    async def broadcast(
        self,
        message_type: str,
        data: Any,
        ttl: int = 3
    ) -> Tuple[List[str], List[str]]:
        """
        广播消息
        :param message_type: 消息类型
        :param data: 消息数据
        :param ttl: 生存时间
        :return: (成功列表, 失败列表)
        """
        try:
            # 构造消息
            message = {
                "type": message_type,
                "data": data,
                "source": self.node_id,
                "ttl": ttl,
                "timestamp": time.time()
            }
            
            # 广播消息
            success, failed = await self.adapter.broadcast(message)
            
            # 更新监控指标
            self.message_counter.inc(
                value=len(success),
                labels={
                    "type": message_type,
                    "direction": "out"
                }
            )
            
            return success, failed
            
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
            raise
            
    def register_message_handler(
        self,
        message_type: str,
        handler: Callable
    ) -> None:
        """
        注册消息处理器
        :param message_type: 消息类型
        :param handler: 处理函数
        """
        if message_type not in self._message_handlers:
            self._message_handlers[message_type] = []
        self._message_handlers[message_type].append(handler)
        
    def unregister_message_handler(
        self,
        message_type: str,
        handler: Callable
    ) -> None:
        """
        注销消息处理器
        :param message_type: 消息类型
        :param handler: 处理函数
        """
        if message_type in self._message_handlers:
            self._message_handlers[message_type].remove(handler)
            
    def get_routing_table(self) -> Dict[str, RoutingTableEntry]:
        """
        获取路由表
        :return: 路由表
        """
        return self._routing_table.copy()
        
    def get_node_info(self, node_id: str) -> Optional[NodeInfo]:
        """
        获取节点信息
        :param node_id: 节点ID
        :return: 节点信息
        """
        if node_id in self._routing_table:
            return self._routing_table[node_id].node
        return None
        
    async def _connect_bootstrap_node(
        self,
        address: str,
        port: int
    ) -> None:
        """
        连接引导节点
        :param address: 地址
        :param port: 端口
        """
        try:
            # 建立连接
            connection = await self.adapter.connect(address, port)
            
            # 发送握手消息
            await self._send_handshake(connection.peer_id)
            
            logger.info(f"Connected to bootstrap node: {address}:{port}")
            
        except Exception as e:
            logger.error(
                f"Failed to connect to bootstrap node {address}:{port}: {e}"
            )
            
    async def _broadcast_leave(self) -> None:
        """广播离开消息"""
        try:
            await self.broadcast(
                "node_leave",
                {
                    "node_id": self.node_id,
                    "timestamp": time.time()
                }
            )
        except Exception as e:
            logger.error(f"Failed to broadcast leave message: {e}")
            
    async def _periodic_refresh(self) -> None:
        """定期刷新"""
        while self.state == NodeState.ONLINE:
            try:
                # 刷新路由表
                await self._refresh_routing_table()
                
                # 清理过期条目
                self._clean_routing_table()
                
                # 更新监控指标
                self.routing_gauge.set(
                    len(self._routing_table),
                    labels={"role": self.role.name.lower()}
                )
                
                # 等待下一次刷新
                await asyncio.sleep(self.refresh_interval)
                
            except Exception as e:
                logger.error(f"Error in periodic refresh: {e}")
                
    async def _refresh_routing_table(self) -> None:
        """刷新路由表"""
        try:
            # 广播ping消息
            await self.broadcast(
                "node_ping",
                {
                    "node_id": self.node_id,
                    "capabilities": list(self.capabilities),
                    "metadata": self.metadata
                }
            )
            
            # 请求邻居节点的路由表
            for node_id in list(self._routing_table.keys()):
                await self._request_routing_table(node_id)
                
        except Exception as e:
            logger.error(f"Failed to refresh routing table: {e}")
            
    def _clean_routing_table(self) -> None:
        """清理路由表"""
        now = time.time()
        expired = []
        
        for node_id, entry in self._routing_table.items():
            # 检查最后可见时间
            if now - entry.last_seen > self.refresh_interval * 3:
                expired.append(node_id)
                continue
                
            # 更新监控指标
            self.reliability_histogram.observe(
                entry.reliability,
                labels={"node_id": node_id}
            )
            
        # 移除过期条目
        for node_id in expired:
            del self._routing_table[node_id]
            
    async def _handle_client_connected(self, event: Any) -> None:
        """
        处理客户端连接
        :param event: 连接事件
        """
        try:
            peer_id = event.data["peer_id"]
            
            # 发送握手消息
            await self._send_handshake(peer_id)
            
            logger.info(f"New peer connected: {peer_id}")
            
        except Exception as e:
            logger.error(f"Error handling client connection: {e}")
            
    async def _handle_client_disconnected(self, event: Any) -> None:
        """
        处理客户端断开
        :param event: 断开事件
        """
        try:
            peer_id = event.data["peer_id"]
            
            # 更新路由表
            if peer_id in self._routing_table:
                del self._routing_table[peer_id]
                
            logger.info(f"Peer disconnected: {peer_id}")
            
        except Exception as e:
            logger.error(f"Error handling client disconnection: {e}")
            
    async def _handle_data_received(self, event: Any) -> None:
        """
        处理接收数据
        :param event: 数据事件
        """
        try:
            peer_id = event.data["peer_id"]
            message = event.data["data"]
            
            # 处理消息
            message_type = message.get("type")
            if message_type in self._message_handlers:
                for handler in self._message_handlers[message_type]:
                    await handler(peer_id, message)
                    
            # 更新监控指标
            self.message_counter.inc(
                labels={
                    "type": message_type,
                    "direction": "in"
                }
            )
            
        except Exception as e:
            logger.error(f"Error handling received data: {e}")
            
    async def _send_handshake(self, peer_id: str) -> None:
        """
        发送握手消息
        :param peer_id: 对端ID
        """
        try:
            # 构造握手消息
            handshake = {
                "type": "handshake",
                "data": {
                    "node_id": self.node_id,
                    "role": self.role.name,
                    "capabilities": list(self.capabilities),
                    "metadata": self.metadata
                }
            }
            
            # 发送消息
            await self.adapter.send(peer_id, handshake)
            
            # 更新监控指标
            self.message_counter.inc(
                labels={
                    "type": "handshake",
                    "direction": "out"
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to send handshake to {peer_id}: {e}")
            
    async def _request_routing_table(self, node_id: str) -> None:
        """
        请求路由表
        :param node_id: 节点ID
        """
        try:
            # 构造请求消息
            request = {
                "type": "routing_table_request",
                "data": {
                    "node_id": self.node_id,
                    "timestamp": time.time()
                }
            }
            
            # 发送请求
            await self.adapter.send(node_id, request)
            
            # 更新监控指标
            self.message_counter.inc(
                labels={
                    "type": "routing_table_request",
                    "direction": "out"
                }
            )
            
        except Exception as e:
            logger.error(
                f"Failed to request routing table from {node_id}: {e}"
            )

def create_p2p_network(
    node_id: str,
    host: str,
    port: int,
    **kwargs
) -> P2PNetwork:
    """
    创建P2P网络
    :param node_id: 节点ID
    :param host: 主机地址
    :param port: 端口
    :param kwargs: 附加参数
    :return: P2P网络实例
    """
    return P2PNetwork(node_id, host, port, **kwargs)
