"""路由模块"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .p2p import NodeInfo, P2PNetwork
from ..monitoring import monitoring

logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    """路由策略"""
    NEAREST = auto()     # 最近节点
    RANDOM = auto()      # 随机节点
    BROADCAST = auto()   # 广播
    LOADBALANCE = auto() # 负载均衡
    REDUNDANT = auto()   # 冗余路由

class MessagePriority(Enum):
    """消息优先级"""
    LOW = 0      # 低优先级
    NORMAL = 1   # 普通优先级
    HIGH = 2     # 高优先级
    URGENT = 3   # 紧急优先级

@dataclass
class RouteInfo:
    """路由信息"""
    source: str                # 源节点
    target: str                # 目标节点
    hops: List[str]           # 路径跳数
    latency: float            # 延迟
    bandwidth: float          # 带宽
    reliability: float        # 可靠性
    timestamp: float          # 时间戳
    metadata: Dict = field(default_factory=dict)  # 元数据

@dataclass
class RoutingMetrics:
    """路由指标"""
    success_count: int = 0        # 成功计数
    failure_count: int = 0        # 失败计数
    total_latency: float = 0.0    # 总延迟
    min_latency: float = float('inf')  # 最小延迟
    max_latency: float = 0.0      # 最大延迟
    last_update: float = 0.0      # 最后更新时间

class MessageRouter:
    """消息路由器"""
    
    def __init__(
        self,
        network: P2PNetwork,
        default_strategy: RoutingStrategy = RoutingStrategy.NEAREST,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.network = network
        self.default_strategy = default_strategy
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 路由表
        self._routes: Dict[str, RouteInfo] = {}
        self._metrics: Dict[str, RoutingMetrics] = {}
        
        # 消息队列
        self._message_queues: Dict[MessagePriority, asyncio.Queue] = {
            priority: asyncio.Queue()
            for priority in MessagePriority
        }
        
        # 监控指标
        self.route_gauge = monitoring.gauge(
            "srp_active_routes",
            "Number of active routes",
            ["strategy"]
        )
        
        self.message_counter = monitoring.counter(
            "srp_routed_messages_total",
            "Total number of routed messages",
            ["status", "strategy"]
        )
        
        self.latency_histogram = monitoring.histogram(
            "srp_message_routing_latency_seconds",
            "Message routing latency in seconds",
            ["target"]
        )
        
        self.retry_counter = monitoring.counter(
            "srp_message_retries_total",
            "Total number of message retries",
            ["target"]
        )
        
        # 启动消息处理
        for priority in MessagePriority:
            asyncio.create_task(
                self._process_message_queue(priority)
            )
            
    async def route_message(
        self,
        target: str,
        message: Any,
        strategy: Optional[RoutingStrategy] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        **kwargs
    ) -> bool:
        """
        路由消息
        :param target: 目标节点
        :param message: 消息内容
        :param strategy: 路由策略
        :param priority: 消息优先级
        :param kwargs: 附加参数
        :return: 是否成功
        """
        try:
            # 选择路由策略
            strategy = strategy or self.default_strategy
            
            # 构造路由消息
            route_message = {
                "target": target,
                "message": message,
                "strategy": strategy,
                "priority": priority,
                "timestamp": time.time(),
                "kwargs": kwargs
            }
            
            # 加入消息队列
            await self._message_queues[priority].put(route_message)
            
            # 更新监控指标
            self.message_counter.inc(
                labels={
                    "status": "queued",
                    "strategy": strategy.name.lower()
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to route message: {e}")
            return False
            
    def update_route(
        self,
        target: str,
        route_info: RouteInfo
    ) -> None:
        """
        更新路由信息
        :param target: 目标节点
        :param route_info: 路由信息
        """
        self._routes[target] = route_info
        
        # 更新指标
        if target not in self._metrics:
            self._metrics[target] = RoutingMetrics()
            
        self._metrics[target].last_update = time.time()
        
        # 更新监控指标
        self.route_gauge.inc(
            labels={
                "strategy": self.default_strategy.name.lower()
            }
        )
        
    def get_route(self, target: str) -> Optional[RouteInfo]:
        """
        获取路由信息
        :param target: 目标节点
        :return: 路由信息
        """
        return self._routes.get(target)
        
    def get_metrics(self, target: str) -> Optional[RoutingMetrics]:
        """
        获取路由指标
        :param target: 目标节点
        :return: 路由指标
        """
        return self._metrics.get(target)
        
    async def _process_message_queue(
        self,
        priority: MessagePriority
    ) -> None:
        """
        处理消息队列
        :param priority: 消息优先级
        """
        while True:
            try:
                # 获取消息
                message = await self._message_queues[priority].get()
                
                # 发送消息
                success = await self._send_message(
                    message["target"],
                    message["message"],
                    message["strategy"],
                    **message["kwargs"]
                )
                
                # 更新监控指标
                self.message_counter.inc(
                    labels={
                        "status": "success" if success else "failed",
                        "strategy": message["strategy"].name.lower()
                    }
                )
                
                # 标记任务完成
                self._message_queues[priority].task_done()
                
            except Exception as e:
                logger.error(f"Error processing message queue: {e}")
                await asyncio.sleep(1)
                
    async def _send_message(
        self,
        target: str,
        message: Any,
        strategy: RoutingStrategy,
        **kwargs
    ) -> bool:
        """
        发送消息
        :param target: 目标节点
        :param message: 消息内容
        :param strategy: 路由策略
        :param kwargs: 附加参数
        :return: 是否成功
        """
        start_time = time.time()
        retries = 0
        success = False
        
        while retries < self.max_retries and not success:
            try:
                # 选择下一跳
                next_hop = await self._select_next_hop(
                    target,
                    strategy,
                    **kwargs
                )
                
                if not next_hop:
                    raise ValueError(f"No route to target: {target}")
                    
                # 发送消息
                success = await self.network.adapter.send(
                    next_hop,
                    message
                )
                
                if success:
                    # 更新指标
                    latency = time.time() - start_time
                    self._update_metrics(target, True, latency)
                    
                    # 更新监控指标
                    self.latency_histogram.observe(
                        latency,
                        labels={"target": target}
                    )
                    
                else:
                    # 更新指标
                    self._update_metrics(target, False, 0)
                    
                    # 更新监控指标
                    self.retry_counter.inc(
                        labels={"target": target}
                    )
                    
                    retries += 1
                    await asyncio.sleep(self.retry_delay)
                    
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
                retries += 1
                await asyncio.sleep(self.retry_delay)
                
        return success
        
    async def _select_next_hop(
        self,
        target: str,
        strategy: RoutingStrategy,
        **kwargs
    ) -> Optional[str]:
        """
        选择下一跳
        :param target: 目标节点
        :param strategy: 路由策略
        :param kwargs: 附加参数
        :return: 下一跳节点ID
        """
        routing_table = self.network.get_routing_table()
        
        if not routing_table:
            return None
            
        if strategy == RoutingStrategy.NEAREST:
            # 选择最近的节点
            return min(
                routing_table.items(),
                key=lambda x: x[1].latency
            )[0]
            
        elif strategy == RoutingStrategy.RANDOM:
            # 随机选择节点
            import random
            return random.choice(list(routing_table.keys()))
            
        elif strategy == RoutingStrategy.BROADCAST:
            # 广播到所有节点
            return list(routing_table.keys())
            
        elif strategy == RoutingStrategy.LOADBALANCE:
            # 负载均衡
            return min(
                routing_table.items(),
                key=lambda x: len(x[1].node.metadata.get("queue", []))
            )[0]
            
        elif strategy == RoutingStrategy.REDUNDANT:
            # 选择多个可靠节点
            reliable_nodes = [
                node_id
                for node_id, entry in routing_table.items()
                if entry.reliability > 0.8
            ]
            return reliable_nodes[:3] if reliable_nodes else None
            
        return None
        
    def _update_metrics(
        self,
        target: str,
        success: bool,
        latency: float
    ) -> None:
        """
        更新路由指标
        :param target: 目标节点
        :param success: 是否成功
        :param latency: 延迟
        """
        if target not in self._metrics:
            self._metrics[target] = RoutingMetrics()
            
        metrics = self._metrics[target]
        
        if success:
            metrics.success_count += 1
            metrics.total_latency += latency
            metrics.min_latency = min(metrics.min_latency, latency)
            metrics.max_latency = max(metrics.max_latency, latency)
        else:
            metrics.failure_count += 1
            
        metrics.last_update = time.time()

def create_message_router(
    network: P2PNetwork,
    **kwargs
) -> MessageRouter:
    """
    创建消息路由器
    :param network: P2P网络实例
    :param kwargs: 附加参数
    :return: 消息路由器实例
    """
    return MessageRouter(network, **kwargs)
