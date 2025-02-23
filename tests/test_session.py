"""
状态管理模块的测试用例
"""

import asyncio
import unittest
import time
from srp.state.session import StateManager, SessionState, DistributedStateManager

class TestStateManager(unittest.TestCase):
    """测试状态管理器"""

    def setUp(self):
        self.manager = StateManager()

    def test_create_session(self):
        """测试创建会话"""
        initial_data = {"user": "alice"}
        session_id = self.manager.create_session(initial_data)
        
        self.assertIsNotNone(session_id)
        state = self.manager.get_state(session_id)
        self.assertEqual(state["user"], "alice")

    def test_update_state(self):
        """测试更新状态"""
        session_id = self.manager.create_session({"count": 0})
        
        # 测试更新
        self.assertTrue(self.manager.update_state(session_id, {"count": 1}))
        state = self.manager.get_state(session_id)
        self.assertEqual(state["count"], 1)
        
        # 测试合并更新
        self.assertTrue(self.manager.update_state(session_id, {"new_key": "value"}, merge=True))
        state = self.manager.get_state(session_id)
        self.assertEqual(state["count"], 1)
        self.assertEqual(state["new_key"], "value")

    def test_delete_session(self):
        """测试删除会话"""
        session_id = self.manager.create_session({"test": "data"})
        
        self.assertTrue(self.manager.delete_session(session_id))
        self.assertIsNone(self.manager.get_state(session_id))

    def test_session_expiration(self):
        """测试会话过期"""
        manager = StateManager(default_ttl=1)  # 1秒后过期
        session_id = manager.create_session({"test": "data"})
        
        # 立即获取应该成功
        self.assertIsNotNone(manager.get_state(session_id))
        
        # 等待过期
        time.sleep(1.1)
        self.assertIsNone(manager.get_state(session_id))

class TestDistributedStateManager(unittest.TestCase):
    """测试分布式状态管理器"""

    def setUp(self):
        self.node1 = DistributedStateManager("node1")
        self.node2 = DistributedStateManager("node2")
        
        # 设置对等节点
        self.node1.peers = {"node2": self.node2}
        self.node2.peers = {"node1": self.node1}

    async def async_test_sync_state(self):
        """测试状态同步"""
        # 在节点1创建会话
        session_id = self.node1.create_session({"data": "from_node1"})
        
        # 同步到节点2
        await self.node2.sync_state("node1", session_id)
        
        # 验证同步结果
        state1 = self.node1.get_state(session_id)
        state2 = self.node2.get_state(session_id)
        
        self.assertEqual(state1["data"], "from_node1")
        self.assertEqual(state2["data"], "from_node1")

    def test_sync_state(self):
        """同步状态的测试包装器"""
        asyncio.run(self.async_test_sync_state())

    def test_version_vector(self):
        """测试版本向量"""
        # 在节点1创建并更新状态
        session_id = self.node1.create_session({"count": 0})
        self.node1.update_state(session_id, {"count": 1})
        
        # 验证版本号增加
        self.assertEqual(self.node1.version_vector["node1"], 1)

if __name__ == '__main__':
    unittest.main()