"""插件系统示例"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional

from srp.plugins.plugin_manager import Plugin, PluginManager
from srp.sdk.client import SRPClient
from srp.network.adapters import LocalAdapter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsPlugin(Plugin):
    """性能指标插件 - 收集和报告系统性能指标"""
    
    __plugin_info__ = {
        "name": "metrics_plugin",
        "version": "0.1.0",
        "description": "System metrics collector",
        "author": "Demo",
        "dependencies": set()
    }
    
    def __init__(self, manager: PluginManager):
        super().__init__(manager)
        self._metrics: Dict[str, float] = {}
        self._collector_task = None
        
    async def on_start(self):
        """启动回调"""
        self.register_hook("get_metrics", self.handle_get_metrics)
        
        # 启动指标收集任务
        self._collector_task = asyncio.create_task(
            self._collect_metrics()
        )
        
        logger.info("Metrics plugin started")
        
    async def on_stop(self):
        """停止回调"""
        if self._collector_task:
            self._collector_task.cancel()
            try:
                await self._collector_task
            except asyncio.CancelledError:
                pass
            self._collector_task = None
            
        logger.info("Metrics plugin stopped")
        
    async def handle_get_metrics(
        self,
        source: str,
        data: Dict,
        context: Optional[Dict] = None
    ) -> bytes:
        """
        处理获取指标请求
        :param source: 消息来源
        :param data: 消息数据
        :param context: 上下文信息
        :return: 响应数据
        """
        # 返回当前指标数据
        return json.dumps({
            "metrics": self._metrics,
            "timestamp": time.time()
        }).encode()
        
    async def _collect_metrics(self):
        """收集系统指标"""
        while True:
            try:
                # 收集CPU使用率
                cpu_percent = await self._get_cpu_usage()
                self._metrics["cpu_usage"] = cpu_percent
                
                # 收集内存使用率
                memory_info = await self._get_memory_info()
                self._metrics.update(memory_info)
                
                # 收集网络指标
                network_info = await self._get_network_info()
                self._metrics.update(network_info)
                
            except Exception as e:
                logger.error(f"Failed to collect metrics: {e}")
                
            await asyncio.sleep(5)  # 每5秒收集一次
            
    async def _get_cpu_usage(self) -> float:
        """
        获取CPU使用率
        :return: CPU使用率
        """
        # 模拟CPU使用率计算
        return 50.0 + (time.time() % 20)
        
    async def _get_memory_info(self) -> Dict[str, float]:
        """
        获取内存信息
        :return: 内存指标
        """
        # 模拟内存信息
        return {
            "memory_usage": 60.0 + (time.time() % 15),
            "swap_usage": 20.0 + (time.time() % 10)
        }
        
    async def _get_network_info(self) -> Dict[str, float]:
        """
        获取网络指标
        :return: 网络指标
        """
        # 模拟网络指标
        return {
            "network_in": 1000 + (time.time() % 500),
            "network_out": 800 + (time.time() % 300)
        }

class LoggingPlugin(Plugin):
    """日志插件 - 处理和过滤日志消息"""
    
    __plugin_info__ = {
        "name": "logging_plugin",
        "version": "0.1.0",
        "description": "Log message handler",
        "author": "Demo",
        "dependencies": set()
    }
    
    def __init__(self, manager: PluginManager):
        super().__init__(manager)
        self._logs: List[Dict] = []
        self._max_logs = 1000
        
    async def on_start(self):
        """启动回调"""
        self.register_hook("log", self.handle_log)
        self.register_hook("get_logs", self.handle_get_logs)
        logger.info("Logging plugin started")
        
    async def on_stop(self):
        """停止回调"""
        logger.info("Logging plugin stopped")
        
    async def handle_log(
        self,
        source: str,
        data: Dict,
        context: Optional[Dict] = None
    ) -> bytes:
        """
        处理日志消息
        :param source: 消息来源
        :param data: 消息数据
        :param context: 上下文信息
        :return: 响应数据
        """
        try:
            # 添加日志记录
            log_entry = {
                "timestamp": time.time(),
                "source": source,
                "level": data.get("level", "INFO"),
                "message": data.get("message", ""),
                "metadata": data.get("metadata", {})
            }
            
            self._logs.append(log_entry)
            
            # 限制日志数量
            if len(self._logs) > self._max_logs:
                self._logs = self._logs[-self._max_logs:]
                
            return json.dumps({
                "status": "success"
            }).encode()
            
        except Exception as e:
            logger.error(f"Failed to handle log: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e)
            }).encode()
            
    async def handle_get_logs(
        self,
        source: str,
        data: Dict,
        context: Optional[Dict] = None
    ) -> bytes:
        """
        处理获取日志请求
        :param source: 消息来源
        :param data: 消息数据
        :param context: 上下文信息
        :return: 响应数据
        """
        try:
            # 过滤参数
            level = data.get("level")
            start_time = data.get("start_time")
            end_time = data.get("end_time")
            limit = data.get("limit", 100)
            
            # 过滤日志
            filtered_logs = self._logs
            
            if level:
                filtered_logs = [
                    log for log in filtered_logs
                    if log["level"] == level
                ]
                
            if start_time:
                filtered_logs = [
                    log for log in filtered_logs
                    if log["timestamp"] >= start_time
                ]
                
            if end_time:
                filtered_logs = [
                    log for log in filtered_logs
                    if log["timestamp"] <= end_time
                ]
                
            # 限制返回数量
            filtered_logs = filtered_logs[-limit:]
            
            return json.dumps({
                "logs": filtered_logs,
                "total": len(filtered_logs)
            }).encode()
            
        except Exception as e:
            logger.error(f"Failed to get logs: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e)
            }).encode()

async def run_example():
    """运行示例"""
    try:
        # 创建客户端
        client = SRPClient(
            node_id="example-node",
            adapter=LocalAdapter("example-node")
        )
        
        # 创建插件管理器
        plugin_manager = PluginManager()
        
        # 注册插件
        metrics_plugin = MetricsPlugin(plugin_manager)
        logging_plugin = LoggingPlugin(plugin_manager)
        
        plugin_manager.add_plugin(metrics_plugin)
        plugin_manager.add_plugin(logging_plugin)
        
        # 启动插件
        await plugin_manager.activate_plugin(metrics_plugin.name)
        await plugin_manager.activate_plugin(logging_plugin.name)
        
        # 启动客户端
        await client.start()
        logger.info("Example node started")
        
        try:
            # 创建会话
            session = await client.create_session({
                "user": "example",
                "role": "admin"
            })
            
            # 发送一些日志
            await client.send_message(
                "example-node",
                message_type="log",
                data={
                    "level": "INFO",
                    "message": "Test log message",
                    "metadata": {
                        "component": "test",
                        "version": "1.0"
                    }
                }
            )
            
            # 获取系统指标
            metrics_response = await client.send_message(
                "example-node",
                message_type="get_metrics",
                data={}
            )
            
            if metrics_response:
                metrics_data = json.loads(metrics_response)
                logger.info(f"System metrics: {metrics_data}")
                
            # 获取日志
            logs_response = await client.send_message(
                "example-node",
                message_type="get_logs",
                data={
                    "level": "INFO",
                    "limit": 10
                }
            )
            
            if logs_response:
                logs_data = json.loads(logs_response)
                logger.info(f"Recent logs: {logs_data}")
                
            # 等待一段时间
            await asyncio.sleep(30)
            
        finally:
            # 关闭会话
            await client.close_session(session.id)
            
            # 停止客户端
            await client.stop()
            
            # 停止插件管理器
            await plugin_manager.stop()
            
            logger.info("Example node stopped")
            
    except Exception as e:
        logger.error(f"Example error: {e}")

def main():
    """主函数"""
    try:
        asyncio.run(run_example())
    except KeyboardInterrupt:
        logger.info("Example stopped by user")

if __name__ == "__main__":
    main()
