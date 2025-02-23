"""
路由模块的完整测试用例
包含安全性、性能和可靠性测试
"""

import unittest
import time
import asyncio
import random
import os
from srp.network.routing import (
    RoutingTable,
    RoutingManager,
    NodeInfo,
    NodeMetrics,
    RoutingError,
    NodeValidationError,
    PathValidationError,
    NetworkPartitionError,
    ReputationManager
)

class TestRoutingTable(unittest.TestCase):
    """测试路由表基本功能"""

    def setUp(self):
        """测试准备"""
        self.table = RoutingTable("0x1234")
        self.test_node = NodeInfo(
            node_id="0x5678",
            address="127.0.0.1",
            port=8001,
            public_key="test_key",
            last_seen=time.time(),
            latency=0.1,
            capacity=0.8,
            is_verified=True,
            metrics=NodeMetrics()
        )

    def test_add_node(self):
        """测试添加节点"""
        success = self.table.add_node(self.test_node)
        self.assertTrue(success)
        self.assertIn(self.test_node.node_id, self.table.node_info)
        
        # 测试重复添加
        success = self.table.add_node(self.test_node)
        self.assertTrue(success)

    def test_remove_node(self):
        """测试移除节点"""
        self.table.add_node(self.test_node)
        success = self.table.remove_node(self.test_node.node_id)
        self.assertTrue(success)
        self.assertNotIn(self.test_node.node_id, self.table.node_info)

    def test_get_bucket_index(self):
        """测试获取桶索引"""
        index = self.table.get_bucket_index("0x5678")
        self.assertIsInstance(index, int)
        self.assertGreaterEqual(index, 0)

    def test_get_closest_nodes(self):
        """测试获取最近节点"""
        # 添加多个测试节点
        test_nodes = [
            NodeInfo(
                node_id=f"0x{i}",
                address="127.0.0.1",
                port=8000+i,
                public_key=f"key_{i}",
                last_seen=time.time(),
                latency=0.1,
                capacity=0.8,
                is_verified=True,
                metrics=NodeMetrics()
            )
            for i in range(5)
        ]
        for node in test_nodes:
            self.table.add_node(node)
            
        closest = self.table.get_closest_nodes("0x0000", limit=3)
        self.assertEqual(len(closest), 3)

    def test_bucket_size_limit(self):
        """测试桶大小限制"""
        # 添加超过桶大小的节点
        for i in range(self.table.bucket_size + 5):
            node = NodeInfo(
                node_id=f"0x{i:04x}",
                address="127.0.0.1",
                port=8000+i,
                public_key=f"key_{i}",
                last_seen=time.time(),
                latency=0.1,
                capacity=0.8,
                is_verified=True,
                metrics=NodeMetrics()
            )
            self.table.add_node(node)
            
        bucket_index = self.table.get_bucket_index("0x0000")
        self.assertLessEqual(
            len(self.table.buckets[bucket_index]),
            self.table.bucket_size
        )

    def test_node_validation(self):
        """测试节点验证"""
        # 创建未验证的节点
        unverified_node = NodeInfo(
            node_id="0xbad",
            address="127.0.0.1",
            port=9999,
            public_key="bad_key",
            last_seen=time.time(),
            latency=0.1,
            capacity=0.8,
            is_verified=False,
            metrics=NodeMetrics()
        )
        
        # 验证失败的节点不应该被添加
        success = self.table.add_node(unverified_node)
        self.assertFalse(success)
        self.assertNotIn(unverified_node.node_id, self.table.node_info)

    def test_node_metrics_update(self):
        """测试节点性能指标更新"""
        self.table.add_node(self.test_node)
        node = self.table.node_info[self.test_node.node_id]
        
        # 更新成功请求
        node.metrics.update(success=True, latency=0.1)
        self.assertEqual(node.metrics.success_count, 1)
        self.assertEqual(node.metrics.error_count, 0)
        self.assertEqual(node.metrics.success_rate, 1.0)
        
        # 更新失败请求
        node.metrics.update(success=False)
        self.assertEqual(node.metrics.success_count, 1)
        self.assertEqual(node.metrics.error_count, 1)
        self.assertEqual(node.metrics.success_rate, 0.5)

class TestReputationSystem(unittest.TestCase):
    """测试信誉系统"""

    def setUp(self):
        self.reputation_manager = ReputationManager()
        self.test_node_id = "0x1234"

    def test_reputation_update(self):
        """测试信誉分数更新"""
        # 初始评分应该是1.0
        self.assertEqual(
            self.reputation_manager.reputation_scores.get(self.test_node_id, 1.0),
            1.0
        )
        
        # 成功请求应该提高信誉
        self.reputation_manager.update_score(
            self.test_node_id,
            success=True,
            response_time=0.1
        )
        new_score = self.reputation_manager.reputation_scores[self.test_node_id]
        self.assertGreater(new_score, 1.0)
        
        # 多次失败应该降低信誉
        for _ in range(5):
            self.reputation_manager.update_score(
                self.test_node_id,
                success=False
            )
        final_score = self.reputation_manager.reputation_scores[self.test_node_id]
        self.assertLess(final_score, new_score)

class TestRoutingManager(unittest.TestCase):
    """测试路由管理器"""

    def setUp(self):
        self.manager = RoutingManager(
            "0x1234",
            secret_key=os.urandom(32),
            max_bandwidth=1e6
        )
        self.test_nodes = [
            NodeInfo(
                node_id=f"0x{i}",
                address="127.0.0.1",
                port=8000+i,
                public_key=f"key_{i}",
                last_seen=time.time(),
                latency=0.1,
                capacity=0.8,
                is_verified=True,
                metrics=NodeMetrics()
            )
            for i in range(5)
        ]

    async def test_node_operations(self):
        """测试节点操作"""
        await self.manager.start()
        
        try:
            # 测试节点更新
            node = self.test_nodes[0]
            await self.manager.update_node(node)
            self.assertIn(
                node.node_id,
                self.manager.routing_table.node_info
            )
            
            # 测试标记节点非活动
            self.manager.mark_node_inactive(node.node_id)
            self.assertFalse(
                self.manager.routing_table.node_info[node.node_id].is_active
            )
            
        finally:
            await self.manager.stop()

    async def test_path_finding(self):
        """测试路径查找"""
        await self.manager.start()
        
        try:
            # 添加测试节点
            for node in self.test_nodes:
                await self.manager.update_node(node)
                
            # 测试基本路径查找
            target_id = self.test_nodes[-1].node_id
            path = await self.manager.find_path(target_id)
            self.assertIsNotNone(path)
            self.assertIn(target_id, path)
            
            # 测试带约束的路径查找
            path = await self.manager.find_path(
                target_id,
                min_success_rate=0.5,
                max_latency=0.5
            )
            if path:
                for node_id in path:
                    node = self.manager.routing_table.node_info[node_id]
                    self.assertGreaterEqual(node.metrics.success_rate, 0.5)
                    self.assertLessEqual(node.latency, 0.5)
                    
        finally:
            await self.manager.stop()

    async def test_network_partition(self):
        """测试网络分区处理"""
        await self.manager.start()
        
        try:
            # 添加节点
            for node in self.test_nodes:
                await self.manager.update_node(node)
                
            # 模拟网络分区
            for node in self.test_nodes[:2]:
                self.manager.mark_node_inactive(node.node_id)
                
            # 检查连通分量
            components = self.manager._find_connected_components()
            self.assertGreater(len(components), 1)
            
            # 测试分区修复
            await self.manager._handle_network_partition(components)
            
            # 验证分区是否被修复
            new_components = self.manager._find_connected_components()
            self.assertEqual(len(new_components), 1)
            
        finally:
            await self.manager.stop()

    async def test_bandwidth_control(self):
        """测试带宽控制"""
        await self.manager.start()
        
        try:
            # 测试正常带宽请求
            allowed = await self.manager.bandwidth_limiter.acquire()
            self.assertTrue(allowed)
            
            # 测试并发请求
            tasks = [
                self.manager.bandwidth_limiter.acquire()
                for _ in range(5)
            ]
            results = await asyncio.gather(*tasks)
            self.assertFalse(all(results))
            
        finally:
            await self.manager.stop()

    async def test_table_optimization(self):
        """测试路由表优化"""
        await self.manager.start()
        
        try:
            # 添加一些重复节点
            for i in range(3):
                for node in self.test_nodes:
                    await self.manager.update_node(node)
                    
            # 执行优化
            await self.manager._optimize_routing_table()
            
            # 检查重复节点是否被移除
            for bucket in self.manager.routing_table.buckets:
                seen = set()
                for node_id in bucket:
                    self.assertNotIn(
                        node_id,
                        seen,
                        "发现重复节点"
                    )
                    seen.add(node_id)
                    
        finally:
            await self.manager.stop()

    async def test_node_validation(self):
        """测试节点验证机制"""
        await self.manager.start()
        
        try:
            # 验证两个节点
            node1, node2 = self.test_nodes[:2]
            await self.manager.update_node(node1)
            await self.manager.update_node(node2)
            
            # 测试互验证
            result = await self.manager._verify_nodes_mutually(
                node1.node_id,
                node2.node_id
            )
            self.assertTrue(result)
            
            # 测试密钥交换
            shared_key = await self.manager.exchange_keys(
                node1.node_id,
                node2.node_id
            )
            self.assertIsNotNone(shared_key)
            
            # 使密钥失效
            self.manager._invalidate_shared_keys(node1.node_id)
            self.assertFalse(
                self.manager.routing_table.node_info[node1.node_id].is_verified
            )
            
        finally:
            await self.manager.stop()

def async_test(coro):
    """异步测试装饰器"""
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))
    return wrapper

if __name__ == '__main__':
    # 配置日志
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    unittest.main()