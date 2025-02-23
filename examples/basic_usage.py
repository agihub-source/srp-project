"""基础使用示例"""

import asyncio
import logging
from pathlib import Path

from srp.network.p2p import create_p2p_network
from srp.sdk.client import SRPClient
from srp.security.encryption import (
    EncryptionAlgorithm,
    SecurityManager,
    create_security_manager
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    try:
        # 创建客户端配置
        client_config = {
            "host": "127.0.0.1",
            "port": 8000,
            "node_id": "example_node_1",
            "bootstrap_nodes": [
                ("127.0.0.1", 8001),
                ("127.0.0.1", 8002)
            ]
        }
        
        # 创建安全管理器
        security = create_security_manager(
            default_algorithm=EncryptionAlgorithm.AES,
            key_size=2048
        )
        
        # 创建P2P网络
        network = await create_p2p_network(
            host=client_config["host"],
            port=client_config["port"],
            node_id=client_config["node_id"],
            security=security
        )
        
        # 创建客户端
        client = SRPClient(
            network=network,
            security=security
        )
        
        # 启动客户端
        await client.start()
        logger.info("Client started")
        
        try:
            # 连接到引导节点
            for host, port in client_config["bootstrap_nodes"]:
                await client.connect_node(host, port)
                
            # 创建会话
            session_id = await client.create_session({
                "user": "alice",
                "type": "example"
            })
            logger.info(f"Created session: {session_id}")
            
            # 发送消息
            response = await client.send_message(
                peer_id="example_node_2",  # 目标节点ID
                message_type="test",      # 消息类型
                data={                    # 消息数据
                    "content": "Hello, SRP!",
                    "timestamp": "2025-02-23T14:30:00Z"
                }
            )
            logger.info(f"Received response: {response}")
            
            # 广播消息
            await client.broadcast_message(
                message_type="announcement",
                data={
                    "content": "Important announcement",
                    "priority": "high"
                }
            )
            
            # 等待一段时间观察网络活动
            await asyncio.sleep(5)
            
            # 获取节点列表
            nodes = await client.list_nodes()
            logger.info(f"Connected nodes: {nodes}")
            
            # 获取路由表
            routes = await client.get_routing_table()
            logger.info(f"Routing table: {routes}")
            
            # 关闭会话
            await client.close_session(session_id)
            logger.info(f"Closed session: {session_id}")
            
        finally:
            # 停止客户端
            await client.stop()
            logger.info("Client stopped")
            
    except Exception as e:
        logger.error(f"Error in example: {e}")
        raise

if __name__ == "__main__":
    # Windows需要使用这种方式运行asyncio
    asyncio.run(main())
