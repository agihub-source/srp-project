"""
P2P网络模块的测试用例
"""

import unittest
import asyncio
import time
from srp.network.p2p import SRPNode, P2PError

class TestSRPNode(unittest.TestCase):
    """测试SRP节点"""

    async def async_setup(self):
        """异步设置"""
        self.node1 = SRPNode(host="127.0.0.1", port=8000)
        self.node2 = SRPNode(host="127.0.0.1", port=8001)
        
        # 启动节点
        await self.node1.start()
        await self.node2.start()

    async def async_cleanup(self):
        """异步清理"""
        await self.node1.stop()
        await self.node2.stop()

    @staticmethod
    def async_test(coro):
        """运行异步测试"""
        return asyncio.run(coro)

    def test_node_initialization(self):
        """测试节点初始化"""
        node = SRPNode()
        self.assertIsNone(node.node)
        self.assertFalse(node._running)
        self.assertIsNotNone(node.state_manager)
        self.assertIsNone(node.routing_manager)

    async def test_start_stop(self):
        """测试节点启动和停止"""
        node = SRPNode()
        
        # 测试启动
        await node.start()
        self.assertTrue(node._running)
        self.assertIsNotNone(node.node)
        self.assertIsNotNone(node.routing_manager)
        
        # 测试停止
        await node.stop()
        self.assertFalse(node._running)

    async def test_connect_peer(self):
        """测试节点连接"""
        await self.async_setup()
        
        # 获取节点2的地址
        node2_addr = f"/ip4/127.0.0.1/tcp/8001/p2p/{self.node2.node.get_id()}"
        
        # 测试连接
        success = await self.node1.connect_peer(node2_addr)
        self.assertTrue(success)
        
        await self.async_cleanup()

    async def test_message_handling(self):
        """测试消息处理"""
        await self.async_setup()
        
        # 创建消息处理器
        received_messages = []
        
        async def test_handler(message):
            received_messages.append(message)
            return {"status": "ok"}
        
        # 注册处理器
        self.node2.register_handler("test", test_handler)
        
        # 连接节点
        node2_addr = f"/ip4/127.0.0.1/tcp/8001/p2p/{self.node2.node.get_id()}"
        await self.node1.connect_peer(node2_addr)
        
        # 发送消息
        test_message = {"content": "Hello, SRP!"}
        response = await self.node1.send_message(
            self.node2.node.get_id(),
            "test",
            test_message
        )
        
        # 验证消息处理
        self.assertEqual(len(received_messages), 1)
        self.assertEqual(received_messages[0].get("content"), "Hello, SRP!")
        self.assertEqual(response.get("status"), "ok")
        
        await self.async_cleanup()

    async def test_protocol_selection(self):
        """测试协议选择"""
        await self.async_setup()
        
        # 测试Protocol Buffers
        pb_message = {
            "model_id": "model_001",
            "context_key": "key_001"
        }
        response = await self.node1.send_message(
            self.node2.node.get_id(),
            "ContextRequest",
            pb_message
        )
        self.assertIsNotNone(response)
        
        # 测试JSON-RPC
        rpc_message = {
            "method": "get_context",
            "params": pb_message
        }
        response = await self.node1.send_message(
            self.node2.node.get_id(),
            "ContextRequest",
            rpc_message,
            use_json_rpc=True
        )
        self.assertIsNotNone(response)
        
        await self.async_cleanup()

    async def test_error_handling(self):
        """测试错误处理"""
        node = SRPNode()
        
        # 测试在未启动时发送消息
        with self.assertRaises(P2PError):
            await node.send_message("invalid_id", "test", {})
        
        # 测试连接到无效节点
        await node.start()
        with self.assertRaises(P2PError):
            await node.connect_peer("/ip4/127.0.0.1/tcp/9999/p2p/invalid_id")
        
        await node.stop()

    async def test_message_handlers(self):
        """测试消息处理器管理"""
        await self.async_setup()
        
        # 测试注册处理器
        async def handler(message):
            return {"echo": message}
        
        self.node1.register_handler("echo", handler)
        self.assertIn("echo", self.node1.message_handlers)
        
        # 测试处理器调用
        test_message = {"test": "data"}
        response = await self.node1.message_handlers["echo"](test_message)
        self.assertEqual(response["echo"], test_message)
        
        await self.async_cleanup()

    async def test_concurrent_messages(self):
        """测试并发消息处理"""
        await self.async_setup()
        
        # 创建计数器
        count = 0
        
        async def slow_handler(message):
            nonlocal count
            await asyncio.sleep(0.1)  # 模拟处理延迟
            count += 1
            return {"count": count}
        
        # 注册处理器
        self.node2.register_handler("slow", slow_handler)
        
        # 连接节点
        node2_addr = f"/ip4/127.0.0.1/tcp/8001/p2p/{self.node2.node.get_id()}"
        await self.node1.connect_peer(node2_addr)
        
        # 并发发送多条消息
        tasks = []
        for _ in range(5):
            task = asyncio.create_task(
                self.node1.send_message(
                    self.node2.node.get_id(),
                    "slow",
                    {"test": "concurrent"}
                )
            )
            tasks.append(task)
        
        # 等待所有消息处理完成
        responses = await asyncio.gather(*tasks)
        
        # 验证所有消息都被处理
        self.assertEqual(count, 5)
        self.assertEqual(len(responses), 5)
        
        await self.async_cleanup()

    async def test_cleanup_task(self):
        """测试清理任务"""
        node = SRPNode()
        await node.start()
        
        # 等待清理任务运行
        await asyncio.sleep(1)
        
        # 验证清理任务在运行
        self.assertIsNotNone(node._cleanup_task)
        self.assertFalse(node._cleanup_task.done())
        
        await node.stop()
        
        # 验证清理任务已停止
        self.assertTrue(node._cleanup_task.cancelled())

def run_async_tests():
    """运行异步测试"""
    test_methods = [
        attr for attr in dir(TestSRPNode)
        if attr.startswith('test_') and asyncio.iscoroutinefunction(
            getattr(TestSRPNode, attr)
        )
    ]
    
    suite = unittest.TestSuite()
    for method in test_methods:
        suite.addTest(TestSRPNode(method))
    
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == '__main__':
    # 运行同步测试
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSRPNode)
    unittest.TextTestRunner().run(suite)
    
    # 运行异步测试
    run_async_tests()