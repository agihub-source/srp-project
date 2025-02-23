"""
通信适配器模块的测试用例
"""

import unittest
from srp.network.adapters import (
    ProtobufAdapter,
    JsonRpcAdapter,
    CommunicationAdapter,
    CommunicationError
)

class TestProtobufAdapter(unittest.TestCase):
    """测试Protocol Buffers适配器"""

    def setUp(self):
        self.adapter = ProtobufAdapter("ContextRequest")

    def test_serialize_deserialize(self):
        """测试序列化和反序列化"""
        test_data = {
            "model_id": "model_001",
            "context_key": "key_001"
        }
        
        # 序列化
        serialized = self.adapter.serialize(test_data)
        self.assertIsInstance(serialized, bytes)
        
        # 反序列化
        deserialized = self.adapter.deserialize(serialized)
        self.assertEqual(deserialized["model_id"], test_data["model_id"])
        self.assertEqual(deserialized["context_key"], test_data["context_key"])

    def test_invalid_message_type(self):
        """测试无效的消息类型"""
        with self.assertRaises(ValueError):
            ProtobufAdapter("InvalidType")

    def test_invalid_data(self):
        """测试无效的数据"""
        with self.assertRaises(CommunicationError):
            self.adapter.serialize({"invalid_field": "value"})

class TestJsonRpcAdapter(unittest.TestCase):
    """测试JSON-RPC适配器"""

    def setUp(self):
        self.adapter = JsonRpcAdapter()

    def test_request_format(self):
        """测试请求格式"""
        test_data = {
            "method": "get_context",
            "params": {"key": "value"}
        }
        
        serialized = self.adapter.serialize(test_data)
        # 验证是否为有效的JSON-RPC请求
        self.assertIn(b'"jsonrpc": "2.0"', serialized)
        self.assertIn(b'"method": "get_context"', serialized)
        self.assertIn(b'"id":', serialized)

    def test_response_handling(self):
        """测试响应处理"""
        response_data = {
            "jsonrpc": "2.0",
            "result": {"key": "value"},
            "id": 1
        }
        
        serialized = str(response_data).encode()
        deserialized = self.adapter.deserialize(serialized)
        self.assertEqual(deserialized, {"key": "value"})

    def test_error_response(self):
        """测试错误响应"""
        error_response = {
            "jsonrpc": "2.0",
            "error": {
                "code": -32600,
                "message": "Invalid Request"
            },
            "id": 1
        }
        
        with self.assertRaises(CommunicationError):
            self.adapter.deserialize(str(error_response).encode())

class TestCommunicationAdapter(unittest.TestCase):
    """测试通信适配器"""

    def setUp(self):
        self.protobuf_comm = CommunicationAdapter(protocol="protobuf")
        self.jsonrpc_comm = CommunicationAdapter(protocol="jsonrpc")

    def test_protocol_selection(self):
        """测试协议选择"""
        # Protocol Buffers适配器
        adapter = self.protobuf_comm.get_adapter("ContextRequest")
        self.assertIsInstance(adapter, ProtobufAdapter)
        
        # JSON-RPC适配器
        adapter = self.jsonrpc_comm.get_adapter("ContextRequest")
        self.assertIsInstance(adapter, JsonRpcAdapter)

    def test_message_handling(self):
        """测试消息处理"""
        test_data = {
            "model_id": "model_001",
            "context_key": "key_001"
        }
        
        # Protocol Buffers
        pb_data = self.protobuf_comm.serialize("ContextRequest", test_data)
        pb_result = self.protobuf_comm.deserialize("ContextRequest", pb_data)
        self.assertEqual(pb_result["model_id"], test_data["model_id"])
        
        # JSON-RPC
        rpc_data = self.jsonrpc_comm.serialize("ContextRequest", {
            "method": "get_context",
            "params": test_data
        })
        # 模拟JSON-RPC响应
        rpc_response = {
            "jsonrpc": "2.0",
            "result": test_data,
            "id": 1
        }
        rpc_result = self.jsonrpc_comm.deserialize(
            "ContextRequest",
            str(rpc_response).encode()
        )
        self.assertEqual(rpc_result["model_id"], test_data["model_id"])

    def test_invalid_protocol(self):
        """测试无效的协议"""
        with self.assertRaises(ValueError):
            CommunicationAdapter(protocol="invalid")

    def test_adapter_caching(self):
        """测试适配器缓存"""
        adapter1 = self.protobuf_comm.get_adapter("ContextRequest")
        adapter2 = self.protobuf_comm.get_adapter("ContextRequest")
        # 验证是否返回相同的适配器实例
        self.assertIs(adapter1, adapter2)

if __name__ == '__main__':
    unittest.main()