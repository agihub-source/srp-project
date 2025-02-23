"""
SRP P2P网络示例
展示了如何构建和管理P2P网络：
- 节点发现和连接
- 路由表管理
- 消息路由
- 网络监控
"""

import asyncio
import random
import logging
from typing import Dict, List, Set
from datetime import datetime

from srp.sdk.client import SRPClient
from srp.network.routing import RoutingManager, NodeInfo, NodeMetrics
from srp.security.encryption import Encryptor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class P2PNode:
    """P2P网络节点"""
    
    def __init__(
        self,
        node_id: str,
        host: str = "127.0.0.1",
        port: int = 8000
    ):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.client = SRPClient(host=host, port=port)
        self.routing_manager = RoutingManager(
            node_id=node_id,
            secret_key=os.urandom(32)
        )
        self.connected_peers: Set[str] = set()
        self.message_handlers: Dict[str, callable] = {}
        self.network_stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_transferred": 0,
            "start_time": datetime.now()
        }
        
    async def start(self):
        """启动节点"""
        await self.client.start()
        await self.routing_manager.start()
        logger.info(f"节点 {self.node_id} 已启动 ({self.host}:{self.port})")
        
    async def stop(self):
        """停止节点"""
        await self.routing_manager.stop()
        await self.client.stop()
        logger.info(f"节点 {self.node_id} 已停止")
        
    async def connect_to_peer(self, peer_id: str, host: str, port: int):
        """连接到对等节点"""
        try:
            # 创建节点信息
            peer_info = NodeInfo(
                node_id=peer_id,
                address=host,
                port=port,
                public_key="",  # 实际应用中应该使用真实的公钥
                last_seen=datetime.now().timestamp(),
                latency=0.1,
                capacity=1.0,
                metrics=NodeMetrics()
            )
            
            # 更新路由表
            await self.routing_manager.update_node(peer_info)
            self.connected_peers.add(peer_id)
            
            logger.info(f"节点 {self.node_id} 已连接到 {peer_id}")
            return True
            
        except Exception as e:
            logger.error(f"连接到节点 {peer_id} 失败: {str(e)}")
            return False
            
    async def broadcast_message(self, message: Dict):
        """广播消息到所有连接的节点"""
        for peer_id in self.connected_peers:
            try:
                await self.send_message(peer_id, message)
            except Exception as e:
                logger.error(f"向节点 {peer_id} 广播消息失败: {str(e)}")
                
    async def send_message(self, target_id: str, message: Dict):
        """发送消息到指定节点"""
        try:
            # 查找路径
            path = await self.routing_manager.find_path(target_id)
            if not path:
                raise Exception(f"无法找到到节点 {target_id} 的路径")
                
            # 准备消息
            message_with_route = {
                "route": path,
                "data": message,
                "timestamp": datetime.now().isoformat()
            }
            
            # 发送消息
            response = await self.client.send_message(
                peer_id=path[0],  # 发送到路径上的下一个节点
                message_type="routed_message",
                data=message_with_route
            )
            
            # 更新统计信息
            self.network_stats["messages_sent"] += 1
            self.network_stats["bytes_transferred"] += len(str(message))
            
            # 更新路径延迟
            if "latency" in response:
                self.routing_manager.update_path_latency(path, response["latency"])
                
            return response
            
        except Exception as e:
            logger.error(f"发送消息到节点 {target_id} 失败: {str(e)}")
            raise
            
    def register_message_handler(self, message_type: str, handler: callable):
        """注册消息处理器"""
        self.message_handlers[message_type] = handler
        
    def get_network_stats(self) -> Dict:
        """获取网络统计信息"""
        stats = self.network_stats.copy()
        stats["uptime"] = (datetime.now() - stats["start_time"]).total_seconds()
        stats["peers_count"] = len(self.connected_peers)
        stats["messages_per_second"] = (
            stats["messages_sent"] + stats["messages_received"]
        ) / stats["uptime"]
        return stats

async def run_p2p_example():
    """运行P2P网络示例"""
    try:
        # 创建多个节点
        nodes: List[P2PNode] = []
        for i in range(5):
            node = P2PNode(
                node_id=f"node_{i}",
                port=8000+i
            )
            await node.start()
            nodes.append(node)
            
        # 建立连接
        # 创建一个简单的网络拓扑：节点0连接到1和2，节点1连接到3，节点2连接到4
        connections = [
            (0, 1), (0, 2),
            (1, 3),
            (2, 4)
        ]
        
        for source, target in connections:
            source_node = nodes[source]
            target_node = nodes[target]
            await source_node.connect_to_peer(
                target_node.node_id,
                target_node.host,
                target_node.port
            )
            
        logger.info("网络拓扑已建立")
        
        # 注册消息处理器
        async def echo_handler(node: P2PNode, message: Dict) -> Dict:
            logger.info(f"节点 {node.node_id} 收到消息: {message}")
            return {
                "status": "success",
                "echo": message,
                "from": node.node_id
            }
            
        for node in nodes:
            node.register_message_handler("echo", echo_handler)
            
        # 发送一些测试消息
        test_messages = [
            {"type": "echo", "content": f"测试消息 {i}"} 
            for i in range(3)
        ]
        
        # 随机选择源节点和目标节点发送消息
        for message in test_messages:
            source = random.choice(nodes)
            target = random.choice([
                n for n in nodes 
                if n.node_id != source.node_id
            ])
            
            logger.info(
                f"从 {source.node_id} 发送消息到 {target.node_id}: "
                f"{message['content']}"
            )
            
            try:
                response = await source.send_message(
                    target.node_id,
                    message
                )
                logger.info(f"收到响应: {response}")
            except Exception as e:
                logger.error(f"发送消息失败: {str(e)}")
                
        # 测试广播
        broadcast_node = nodes[0]
        broadcast_message = {
            "type": "broadcast",
            "content": "这是一条广播消息"
        }
        
        logger.info(f"节点 {broadcast_node.node_id} 广播消息")
        await broadcast_node.broadcast_message(broadcast_message)
        
        # 显示网络统计
        for node in nodes:
            stats = node.get_network_stats()
            logger.info(f"\n节点 {node.node_id} 统计信息:")
            logger.info(f"- 运行时间: {stats['uptime']:.1f} 秒")
            logger.info(f"- 连接节点数: {stats['peers_count']}")
            logger.info(f"- 发送消息数: {stats['messages_sent']}")
            logger.info(f"- 接收消息数: {stats['messages_received']}")
            logger.info(f"- 传输字节数: {stats['bytes_transferred']}")
            logger.info(f"- 消息频率: {stats['messages_per_second']:.2f} msg/s")
            
        # 等待一段时间以便查看网络活动
        await asyncio.sleep(5)
        
    finally:
        # 停止所有节点
        for node in nodes:
            await node.stop()

def main():
    """主函数"""
    try:
        asyncio.run(run_p2p_example())
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序异常退出: {str(e)}")

if __name__ == "__main__":
    import os  # 添加缺失的导入
    main()