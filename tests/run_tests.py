"""
SRP测试运行器 - 运行所有测试用例
"""

import unittest
import asyncio
import sys
from test_session import TestStateManager, TestDistributedStateManager
from test_adapters import TestProtobufAdapter, TestJsonRpcAdapter, TestCommunicationAdapter
from test_routing import TestRoutingTable, TestRoutingManager
from test_p2p import TestSRPNode, run_async_tests

def run_all_tests():
    """运行所有测试用例"""
    # 创建测试加载器
    loader = unittest.TestLoader()
    
    # 创建测试套件
    suite = unittest.TestSuite()
    
    # 添加同步测试
    sync_test_cases = [
        TestStateManager,
        TestProtobufAdapter,
        TestJsonRpcAdapter,
        TestCommunicationAdapter,
        TestRoutingTable
    ]
    
    for test_case in sync_test_cases:
        suite.addTests(loader.loadTestsFromTestCase(test_case))
    
    # 运行同步测试
    print("\n=== 运行同步测试 ===")
    runner = unittest.TextTestRunner(verbosity=2)
    sync_result = runner.run(suite)
    
    # 运行异步测试
    print("\n=== 运行异步测试 ===")
    async_test_cases = [
        TestDistributedStateManager,
        TestRoutingManager,
        TestSRPNode
    ]
    
    async_results = []
    for test_case in async_test_cases:
        # 获取所有异步测试方法
        async_methods = [
            attr for attr in dir(test_case)
            if attr.startswith('test_') and asyncio.iscoroutinefunction(
                getattr(test_case, attr)
            )
        ]
        
        # 为每个异步测试方法创建测试实例
        async_suite = unittest.TestSuite()
        for method in async_methods:
            async_suite.addTest(test_case(method))
        
        # 运行异步测试套件
        async_result = runner.run(async_suite)
        async_results.append(async_result)
    
    # 输出测试统计
    total_tests = (
        sync_result.testsRun +
        sum(result.testsRun for result in async_results)
    )
    total_failures = (
        len(sync_result.failures) +
        sum(len(result.failures) for result in async_results)
    )
    total_errors = (
        len(sync_result.errors) +
        sum(len(result.errors) for result in async_results)
    )
    
    print("\n=== 测试统计 ===")
    print(f"总测试数: {total_tests}")
    print(f"失败数: {total_failures}")
    print(f"错误数: {total_errors}")
    
    # 如果有任何测试失败，返回非零退出码
    if total_failures > 0 or total_errors > 0:
        return 1
    return 0

if __name__ == '__main__':
    try:
        exit_code = run_all_tests()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n运行测试时发生错误: {e}")
        sys.exit(1)