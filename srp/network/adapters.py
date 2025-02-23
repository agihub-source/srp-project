"""网络适配器模块"""

import asyncio
import logging
import socket
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..monitoring import monitoring
from ..state.session import Session, session_manager

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    """连接状态"""
    DISCONNECTED = auto()  # 未连接
    CONNECTING = auto()    # 连接中
    CONNECTED = auto()     # 已连接
    CLOSING = auto()       # 关闭中
    ERROR = auto()         # 错误

@dataclass
class ConnectionInfo:
    """连接信息"""
    peer_id: str              # 对端ID
    address: str              # 地址
    port: int                # 端口
    protocol: str            # 协议
    state: ConnectionState   # 状态
    latency: float = 0.0     # 延迟(秒)
    bandwidth: float = 0.0   # 带宽(bytes/s)
    connected_at: float = 0  # 连接时间戳
    session_id: Optional[str] = None  # 会话ID

class NetworkEvent:
    """网络事件"""
    
    def __init__(self, event_type: str, **kwargs):
        self.type = event_type
        self.timestamp = asyncio.get_event_loop().time()
        self.data = kwargs

class NetworkAdapter(ABC):
    """网络适配器基类"""
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._connections: Dict[str, ConnectionInfo] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        # 监控指标
        self.connection_gauge = monitoring.gauge(
            "srp_active_connections",
            "Number of active connections",
            ["state", "protocol"]
        )
        
        self.bandwidth_histogram = monitoring.histogram(
            "srp_bandwidth_bytes",
            "Connection bandwidth in bytes per second",
            ["direction", "peer_id"]
        )
        
        self.latency_histogram = monitoring.histogram(
            "srp_latency_seconds",
            "Connection latency in seconds",
            ["peer_id"]
        )
        
        self.error_counter = monitoring.counter(
            "srp_network_errors_total",
            "Total number of network errors",
            ["type"]
        )
        
    @abstractmethod
    async def start(self) -> None:
        """启动适配器"""
        pass
        
    @abstractmethod
    async def stop(self) -> None:
        """停止适配器"""
        pass
        
    @abstractmethod
    async def connect(self, address: str, port: int) -> ConnectionInfo:
        """
        连接到对端
        :param address: 地址
        :param port: 端口
        :return: 连接信息
        """
        pass
        
    @abstractmethod
    async def disconnect(self, peer_id: str) -> bool:
        """
        断开与对端的连接
        :param peer_id: 对端ID
        :return: 是否成功
        """
        pass
        
    @abstractmethod
    async def send(
        self,
        peer_id: str,
        data: Union[str, bytes],
        **kwargs
    ) -> bool:
        """
        发送数据
        :param peer_id: 对端ID
        :param data: 数据
        :param kwargs: 附加参数
        :return: 是否成功
        """
        pass
        
    @abstractmethod
    async def broadcast(
        self,
        data: Union[str, bytes],
        **kwargs
    ) -> Tuple[List[str], List[str]]:
        """
        广播数据
        :param data: 数据
        :param kwargs: 附加参数
        :return: (成功列表, 失败列表)
        """
        pass
        
    def register_event_handler(
        self,
        event_type: str,
        handler: Callable
    ) -> None:
        """
        注册事件处理器
        :param event_type: 事件类型
        :param handler: 处理函数
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        
    def unregister_event_handler(
        self,
        event_type: str,
        handler: Callable
    ) -> None:
        """
        注销事件处理器
        :param event_type: 事件类型
        :param handler: 处理函数
        """
        if event_type in self._event_handlers:
            self._event_handlers[event_type].remove(handler)
            
    async def _emit_event(self, event: NetworkEvent) -> None:
        """
        触发事件
        :param event: 事件对象
        """
        if event.type in self._event_handlers:
            for handler in self._event_handlers[event.type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(
                        f"Error in event handler for {event.type}: {e}"
                    )
                    self.error_counter.inc(
                        labels={"type": "event_handler"}
                    )
                    
    def _update_connection_state(
        self,
        peer_id: str,
        state: ConnectionState
    ) -> None:
        """
        更新连接状态
        :param peer_id: 对端ID
        :param state: 新状态
        """
        if peer_id in self._connections:
            old_state = self._connections[peer_id].state
            self._connections[peer_id].state = state
            
            # 更新监控指标
            self.connection_gauge.dec(
                labels={
                    "state": old_state.name.lower(),
                    "protocol": self._connections[peer_id].protocol
                }
            )
            self.connection_gauge.inc(
                labels={
                    "state": state.name.lower(),
                    "protocol": self._connections[peer_id].protocol
                }
            )
            
    def _update_connection_metrics(
        self,
        peer_id: str,
        latency: float = None,
        bandwidth: float = None
    ) -> None:
        """
        更新连接指标
        :param peer_id: 对端ID
        :param latency: 延迟(秒)
        :param bandwidth: 带宽(bytes/s)
        """
        if peer_id in self._connections:
            if latency is not None:
                self._connections[peer_id].latency = latency
                self.latency_histogram.observe(
                    latency,
                    labels={"peer_id": peer_id}
                )
                
            if bandwidth is not None:
                self._connections[peer_id].bandwidth = bandwidth
                self.bandwidth_histogram.observe(
                    bandwidth,
                    labels={
                        "direction": "in",
                        "peer_id": peer_id
                    }
                )

class TCPAdapter(NetworkAdapter):
    """TCP适配器"""
    
    def __init__(
        self,
        host: str,
        port: int,
        backlog: int = 100,
        timeout: float = 30.0
    ):
        super().__init__(host, port)
        self.backlog = backlog
        self.timeout = timeout
        self._server: Optional[asyncio.Server] = None
        self._clients: Dict[str, asyncio.Transport] = {}
        
    async def start(self) -> None:
        """启动适配器"""
        try:
            self._server = await asyncio.start_server(
                self._handle_client,
                self.host,
                self.port,
                backlog=self.backlog
            )
            logger.info(f"TCP server started on {self.host}:{self.port}")
            
            await self._server.start_serving()
            
        except Exception as e:
            logger.error(f"Failed to start TCP server: {e}")
            self.error_counter.inc(labels={"type": "server_start"})
            raise
            
    async def stop(self) -> None:
        """停止适配器"""
        try:
            if self._server:
                self._server.close()
                await self._server.wait_closed()
                
            for peer_id, transport in self._clients.items():
                transport.close()
                await self.disconnect(peer_id)
                
            logger.info("TCP server stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop TCP server: {e}")
            self.error_counter.inc(labels={"type": "server_stop"})
            
    async def connect(self, address: str, port: int) -> ConnectionInfo:
        """连接到对端"""
        try:
            # 创建连接
            loop = asyncio.get_event_loop()
            transport, _ = await loop.create_connection(
                lambda: self._create_client_protocol(),
                address,
                port
            )
            
            # 创建连接信息
            peer_id = f"{address}:{port}"
            connection = ConnectionInfo(
                peer_id=peer_id,
                address=address,
                port=port,
                protocol="tcp",
                state=ConnectionState.CONNECTED,
                connected_at=loop.time()
            )
            
            self._connections[peer_id] = connection
            self._clients[peer_id] = transport
            
            # 更新监控指标
            self.connection_gauge.inc(
                labels={
                    "state": "connected",
                    "protocol": "tcp"
                }
            )
            
            logger.info(f"Connected to {address}:{port}")
            return connection
            
        except Exception as e:
            logger.error(f"Failed to connect to {address}:{port}: {e}")
            self.error_counter.inc(labels={"type": "connect"})
            raise
            
    async def disconnect(self, peer_id: str) -> bool:
        """断开连接"""
        try:
            if peer_id not in self._clients:
                return False
                
            # 关闭传输
            self._clients[peer_id].close()
            del self._clients[peer_id]
            
            # 更新状态
            self._update_connection_state(
                peer_id,
                ConnectionState.DISCONNECTED
            )
            
            # 清理连接信息
            if peer_id in self._connections:
                del self._connections[peer_id]
                
            logger.info(f"Disconnected from {peer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to disconnect from {peer_id}: {e}")
            self.error_counter.inc(labels={"type": "disconnect"})
            return False
            
    async def send(
        self,
        peer_id: str,
        data: Union[str, bytes],
        **kwargs
    ) -> bool:
        """发送数据"""
        try:
            if peer_id not in self._clients:
                return False
                
            # 编码数据
            if isinstance(data, str):
                data = data.encode()
                
            # 发送数据
            self._clients[peer_id].write(data)
            await self._clients[peer_id].drain()
            
            # 更新带宽指标
            self._update_connection_metrics(
                peer_id,
                bandwidth=len(data)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send data to {peer_id}: {e}")
            self.error_counter.inc(labels={"type": "send"})
            return False
            
    async def broadcast(
        self,
        data: Union[str, bytes],
        **kwargs
    ) -> Tuple[List[str], List[str]]:
        """广播数据"""
        success = []
        failed = []
        
        for peer_id in self._clients:
            if await self.send(peer_id, data, **kwargs):
                success.append(peer_id)
            else:
                failed.append(peer_id)
                
        return success, failed
        
    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter
    ) -> None:
        """
        处理客户端连接
        :param reader: 读取器
        :param writer: 写入器
        """
        peer_name = writer.get_extra_info('peername')
        peer_id = f"{peer_name[0]}:{peer_name[1]}"
        
        try:
            # 创建连接信息
            connection = ConnectionInfo(
                peer_id=peer_id,
                address=peer_name[0],
                port=peer_name[1],
                protocol="tcp",
                state=ConnectionState.CONNECTED,
                connected_at=asyncio.get_event_loop().time()
            )
            
            self._connections[peer_id] = connection
            self._clients[peer_id] = writer
            
            # 更新监控指标
            self.connection_gauge.inc(
                labels={
                    "state": "connected",
                    "protocol": "tcp"
                }
            )
            
            logger.info(f"New client connected: {peer_id}")
            
            # 触发连接事件
            await self._emit_event(
                NetworkEvent(
                    "client_connected",
                    peer_id=peer_id,
                    connection=connection
                )
            )
            
            # 读取数据
            while True:
                try:
                    data = await reader.read(8192)
                    if not data:
                        break
                        
                    # 更新带宽指标
                    self._update_connection_metrics(
                        peer_id,
                        bandwidth=len(data)
                    )
                    
                    # 触发数据接收事件
                    await self._emit_event(
                        NetworkEvent(
                            "data_received",
                            peer_id=peer_id,
                            data=data
                        )
                    )
                    
                except asyncio.CancelledError:
                    break
                    
                except Exception as e:
                    logger.error(
                        f"Error reading from client {peer_id}: {e}"
                    )
                    self.error_counter.inc(
                        labels={"type": "read"}
                    )
                    break
                    
            # 关闭连接
            await self.disconnect(peer_id)
            
            # 触发断开事件
            await self._emit_event(
                NetworkEvent(
                    "client_disconnected",
                    peer_id=peer_id
                )
            )
            
        except Exception as e:
            logger.error(f"Error handling client {peer_id}: {e}")
            self.error_counter.inc(labels={"type": "handler"})
            
    def _create_client_protocol(self) -> asyncio.Protocol:
        """创建客户端协议"""
        return asyncio.Protocol()

class UDPAdapter(NetworkAdapter):
    """UDP适配器"""
    
    def __init__(
        self,
        host: str,
        port: int,
        ttl: int = 1,
        buffer_size: int = 65507
    ):
        super().__init__(host, port)
        self.ttl = ttl
        self.buffer_size = buffer_size
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._protocol: Optional[asyncio.DatagramProtocol] = None
        
    async def start(self) -> None:
        """启动适配器"""
        try:
            loop = asyncio.get_event_loop()
            self._transport, self._protocol = await loop.create_datagram_endpoint(
                lambda: self._create_datagram_protocol(),
                local_addr=(self.host, self.port)
            )
            
            # 设置TTL
            sock = self._transport.get_extra_info('socket')
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_TTL, self.ttl)
            
            logger.info(f"UDP server started on {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start UDP server: {e}")
            self.error_counter.inc(labels={"type": "server_start"})
            raise
            
    async def stop(self) -> None:
        """停止适配器"""
        try:
            if self._transport:
                self._transport.close()
                
            logger.info("UDP server stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop UDP server: {e}")
            self.error_counter.inc(labels={"type": "server_stop"})
            
    async def connect(self, address: str, port: int) -> ConnectionInfo:
        """连接到对端"""
        try:
            peer_id = f"{address}:{port}"
            connection = ConnectionInfo(
                peer_id=peer_id,
                address=address,
                port=port,
                protocol="udp",
                state=ConnectionState.CONNECTED,
                connected_at=asyncio.get_event_loop().time()
            )
            
            self._connections[peer_id] = connection
            
            # 更新监控指标
            self.connection_gauge.inc(
                labels={
                    "state": "connected",
                    "protocol": "udp"
                }
            )
            
            logger.info(f"Connected to {address}:{port}")
            return connection
            
        except Exception as e:
            logger.error(f"Failed to connect to {address}:{port}: {e}")
            self.error_counter.inc(labels={"type": "connect"})
            raise
            
    async def disconnect(self, peer_id: str) -> bool:
        """断开连接"""
        try:
            if peer_id not in self._connections:
                return False
                
            # 更新状态
            self._update_connection_state(
                peer_id,
                ConnectionState.DISCONNECTED
            )
            
            # 清理连接信息
            del self._connections[peer_id]
            
            logger.info(f"Disconnected from {peer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to disconnect from {peer_id}: {e}")
            self.error_counter.inc(labels={"type": "disconnect"})
            return False
            
    async def send(
        self,
        peer_id: str,
        data: Union[str, bytes],
        **kwargs
    ) -> bool:
        """发送数据"""
        try:
            if peer_id not in self._connections:
                return False
                
            connection = self._connections[peer_id]
            
            # 编码数据
            if isinstance(data, str):
                data = data.encode()
                
            # 发送数据
            self._transport.sendto(
                data,
                (connection.address, connection.port)
            )
            
            # 更新带宽指标
            self._update_connection_metrics(
                peer_id,
                bandwidth=len(data)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send data to {peer_id}: {e}")
            self.error_counter.inc(labels={"type": "send"})
            return False
            
    async def broadcast(
        self,
        data: Union[str, bytes],
        **kwargs
    ) -> Tuple[List[str], List[str]]:
        """广播数据"""
        success = []
        failed = []
        
        for peer_id in self._connections:
            if await self.send(peer_id, data, **kwargs):
                success.append(peer_id)
            else:
                failed.append(peer_id)
                
        return success, failed
        
    def _create_datagram_protocol(self) -> asyncio.DatagramProtocol:
        """创建数据报协议"""
        return DatagramProtocol(self)

class DatagramProtocol(asyncio.DatagramProtocol):
    """数据报协议"""
    
    def __init__(self, adapter: UDPAdapter):
        self.adapter = adapter
        
    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        """
        接收数据报
        :param data: 数据
        :param addr: 地址
        """
        peer_id = f"{addr[0]}:{addr[1]}"
        
        try:
            # 更新带宽指标
            self.adapter._update_connection_metrics(
                peer_id,
                bandwidth=len(data)
            )
            
            # 触发数据接收事件
            asyncio.create_task(
                self.adapter._emit_event(
                    NetworkEvent(
                        "data_received",
                        peer_id=peer_id,
                        data=data
                    )
                )
            )
            
        except Exception as e:
            logger.error(f"Error handling datagram from {peer_id}: {e}")
            self.adapter.error_counter.inc(
                labels={"type": "datagram"}
            )
