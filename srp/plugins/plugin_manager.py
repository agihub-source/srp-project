"""插件系统模块"""

import asyncio
import importlib
import inspect
import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union

from ..monitoring import monitoring

logger = logging.getLogger(__name__)

class PluginState(Enum):
    """插件状态"""
    UNLOADED = auto()    # 未加载
    LOADED = auto()      # 已加载
    ENABLED = auto()     # 已启用
    DISABLED = auto()    # 已禁用
    ERROR = auto()       # 错误状态

class PluginPriority(Enum):
    """插件优先级"""
    LOW = 0     # 低优先级
    NORMAL = 1  # 普通优先级
    HIGH = 2    # 高优先级
    SYSTEM = 3  # 系统级

@dataclass
class PluginInfo:
    """插件信息"""
    name: str                    # 插件名称
    version: str                # 插件版本
    description: str            # 插件描述
    author: str                # 插件作者
    dependencies: List[str]    # 依赖项
    entry_point: str           # 入口点
    priority: PluginPriority   # 优先级
    state: PluginState        # 状态
    metadata: Dict = field(default_factory=dict)  # 元数据

@dataclass
class PluginContext:
    """插件上下文"""
    plugin_info: PluginInfo    # 插件信息
    config: Dict              # 配置信息
    data: Dict = field(default_factory=dict)  # 数据存储
    events: Dict = field(default_factory=dict) # 事件处理器

class Plugin:
    """插件基类"""
    
    def __init__(self, context: PluginContext):
        self.context = context
        self.logger = logging.getLogger(
            f"srp.plugin.{context.plugin_info.name}"
        )
        
    async def on_load(self) -> None:
        """加载时调用"""
        pass
        
    async def on_enable(self) -> None:
        """启用时调用"""
        pass
        
    async def on_disable(self) -> None:
        """禁用时调用"""
        pass
        
    async def on_unload(self) -> None:
        """卸载时调用"""
        pass
        
    async def handle_event(self, event_type: str, data: Any) -> None:
        """
        处理事件
        :param event_type: 事件类型
        :param data: 事件数据
        """
        handler = self.context.events.get(event_type)
        if handler:
            await handler(data)
            
    def register_event_handler(
        self,
        event_type: str,
        handler: callable
    ) -> None:
        """
        注册事件处理器
        :param event_type: 事件类型
        :param handler: 处理函数
        """
        self.context.events[event_type] = handler
        
    def unregister_event_handler(self, event_type: str) -> None:
        """
        注销事件处理器
        :param event_type: 事件类型
        """
        self.context.events.pop(event_type, None)

class PluginManager:
    """插件管理器"""
    
    def __init__(
        self,
        plugin_dir: Union[str, Path],
        config_dir: Optional[Union[str, Path]] = None
    ):
        self.plugin_dir = Path(plugin_dir)
        self.config_dir = Path(config_dir) if config_dir else None
        
        # 插件注册表
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_infos: Dict[str, PluginInfo] = {}
        
        # 监控指标
        self.plugin_gauge = monitoring.gauge(
            "srp_plugins",
            "Number of plugins",
            ["state"]
        )
        
        self.event_counter = monitoring.counter(
            "srp_plugin_events_total",
            "Total number of plugin events",
            ["plugin", "event_type"]
        )
        
        self.error_counter = monitoring.counter(
            "srp_plugin_errors_total",
            "Total number of plugin errors",
            ["plugin", "error_type"]
        )
        
        # 创建插件目录
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        if self.config_dir:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
    async def load_plugins(self) -> None:
        """加载所有插件"""
        try:
            # 获取插件列表
            plugin_paths = list(self.plugin_dir.glob("*/plugin.json"))
            
            # 按优先级排序加载
            for plugin_path in sorted(
                plugin_paths,
                key=lambda p: self._get_plugin_priority(p)
            ):
                await self.load_plugin(plugin_path.parent)
                
        except Exception as e:
            logger.error(f"Failed to load plugins: {e}")
            raise
            
    async def load_plugin(self, plugin_path: Union[str, Path]) -> None:
        """
        加载插件
        :param plugin_path: 插件路径
        """
        try:
            plugin_path = Path(plugin_path)
            manifest_path = plugin_path / "plugin.json"
            
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Plugin manifest not found: {manifest_path}"
                )
                
            # 读取插件信息
            import json
            with open(manifest_path) as f:
                manifest = json.load(f)
                
            # 创建插件信息
            plugin_info = PluginInfo(
                name=manifest["name"],
                version=manifest["version"],
                description=manifest.get("description", ""),
                author=manifest.get("author", ""),
                dependencies=manifest.get("dependencies", []),
                entry_point=manifest["entry_point"],
                priority=PluginPriority[
                    manifest.get("priority", "NORMAL").upper()
                ],
                state=PluginState.UNLOADED,
                metadata=manifest.get("metadata", {})
            )
            
            # 检查依赖
            await self._check_dependencies(plugin_info)
            
            # 加载插件模块
            sys.path.insert(0, str(plugin_path))
            module = importlib.import_module(plugin_info.entry_point)
            plugin_class = self._find_plugin_class(module)
            
            # 加载配置
            config = await self._load_plugin_config(plugin_info.name)
            
            # 创建上下文
            context = PluginContext(
                plugin_info=plugin_info,
                config=config
            )
            
            # 创建插件实例
            plugin = plugin_class(context)
            
            # 调用加载回调
            await plugin.on_load()
            
            # 注册插件
            self._plugins[plugin_info.name] = plugin
            self._plugin_infos[plugin_info.name] = plugin_info
            plugin_info.state = PluginState.LOADED
            
            # 更新监控指标
            self.plugin_gauge.inc(
                labels={"state": plugin_info.state.name.lower()}
            )
            
            logger.info(f"Loaded plugin: {plugin_info.name}")
            
        except Exception as e:
            logger.error(f"Failed to load plugin: {e}")
            self.error_counter.inc(
                labels={
                    "plugin": plugin_info.name,
                    "error_type": "load"
                }
            )
            raise
            
    async def enable_plugin(self, name: str) -> None:
        """
        启用插件
        :param name: 插件名称
        """
        try:
            plugin = self._plugins.get(name)
            if not plugin:
                raise ValueError(f"Plugin not found: {name}")
                
            plugin_info = self._plugin_infos[name]
            if plugin_info.state != PluginState.LOADED:
                raise ValueError(
                    f"Plugin {name} is not in LOADED state"
                )
                
            # 调用启用回调
            await plugin.on_enable()
            
            # 更新状态
            plugin_info.state = PluginState.ENABLED
            
            # 更新监控指标
            self.plugin_gauge.inc(
                labels={"state": PluginState.ENABLED.name.lower()}
            )
            self.plugin_gauge.dec(
                labels={"state": PluginState.LOADED.name.lower()}
            )
            
            logger.info(f"Enabled plugin: {name}")
            
        except Exception as e:
            logger.error(f"Failed to enable plugin: {e}")
            self.error_counter.inc(
                labels={
                    "plugin": name,
                    "error_type": "enable"
                }
            )
            raise
            
    async def disable_plugin(self, name: str) -> None:
        """
        禁用插件
        :param name: 插件名称
        """
        try:
            plugin = self._plugins.get(name)
            if not plugin:
                raise ValueError(f"Plugin not found: {name}")
                
            plugin_info = self._plugin_infos[name]
            if plugin_info.state != PluginState.ENABLED:
                raise ValueError(
                    f"Plugin {name} is not in ENABLED state"
                )
                
            # 调用禁用回调
            await plugin.on_disable()
            
            # 更新状态
            plugin_info.state = PluginState.DISABLED
            
            # 更新监控指标
            self.plugin_gauge.inc(
                labels={"state": PluginState.DISABLED.name.lower()}
            )
            self.plugin_gauge.dec(
                labels={"state": PluginState.ENABLED.name.lower()}
            )
            
            logger.info(f"Disabled plugin: {name}")
            
        except Exception as e:
            logger.error(f"Failed to disable plugin: {e}")
            self.error_counter.inc(
                labels={
                    "plugin": name,
                    "error_type": "disable"
                }
            )
            raise
            
    async def unload_plugin(self, name: str) -> None:
        """
        卸载插件
        :param name: 插件名称
        """
        try:
            plugin = self._plugins.get(name)
            if not plugin:
                raise ValueError(f"Plugin not found: {name}")
                
            plugin_info = self._plugin_infos[name]
            
            # 如果插件已启用，先禁用
            if plugin_info.state == PluginState.ENABLED:
                await self.disable_plugin(name)
                
            # 调用卸载回调
            await plugin.on_unload()
            
            # 移除插件
            del self._plugins[name]
            del self._plugin_infos[name]
            
            # 更新监控指标
            self.plugin_gauge.dec(
                labels={"state": plugin_info.state.name.lower()}
            )
            
            logger.info(f"Unloaded plugin: {name}")
            
        except Exception as e:
            logger.error(f"Failed to unload plugin: {e}")
            self.error_counter.inc(
                labels={
                    "plugin": name,
                    "error_type": "unload"
                }
            )
            raise
            
    async def reload_plugin(self, name: str) -> None:
        """
        重新加载插件
        :param name: 插件名称
        """
        try:
            plugin = self._plugins.get(name)
            if not plugin:
                raise ValueError(f"Plugin not found: {name}")
                
            # 获取插件路径
            plugin_path = self.plugin_dir / name
            
            # 卸载插件
            await self.unload_plugin(name)
            
            # 重新加载插件
            await self.load_plugin(plugin_path)
            
            logger.info(f"Reloaded plugin: {name}")
            
        except Exception as e:
            logger.error(f"Failed to reload plugin: {e}")
            self.error_counter.inc(
                labels={
                    "plugin": name,
                    "error_type": "reload"
                }
            )
            raise
            
    async def broadcast_event(
        self,
        event_type: str,
        data: Any
    ) -> None:
        """
        广播事件
        :param event_type: 事件类型
        :param data: 事件数据
        """
        try:
            for name, plugin in self._plugins.items():
                if self._plugin_infos[name].state == PluginState.ENABLED:
                    try:
                        await plugin.handle_event(event_type, data)
                        
                        # 更新监控指标
                        self.event_counter.inc(
                            labels={
                                "plugin": name,
                                "event_type": event_type
                            }
                        )
                        
                    except Exception as e:
                        logger.error(
                            f"Failed to handle event {event_type} "
                            f"in plugin {name}: {e}"
                        )
                        self.error_counter.inc(
                            labels={
                                "plugin": name,
                                "error_type": "event_handler"
                            }
                        )
                        
        except Exception as e:
            logger.error(f"Failed to broadcast event: {e}")
            raise
            
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """
        获取插件实例
        :param name: 插件名称
        :return: 插件实例
        """
        return self._plugins.get(name)
        
    def get_plugin_info(self, name: str) -> Optional[PluginInfo]:
        """
        获取插件信息
        :param name: 插件名称
        :return: 插件信息
        """
        return self._plugin_infos.get(name)
        
    def list_plugins(self) -> List[PluginInfo]:
        """
        获取插件列表
        :return: 插件信息列表
        """
        return list(self._plugin_infos.values())
        
    async def _check_dependencies(self, plugin_info: PluginInfo) -> None:
        """
        检查依赖
        :param plugin_info: 插件信息
        """
        for dep in plugin_info.dependencies:
            if dep not in self._plugins:
                raise ValueError(
                    f"Missing dependency {dep} for plugin {plugin_info.name}"
                )
                
    async def _load_plugin_config(self, name: str) -> Dict:
        """
        加载插件配置
        :param name: 插件名称
        :return: 配置信息
        """
        if not self.config_dir:
            return {}
            
        config_path = self.config_dir / f"{name}.json"
        if not config_path.exists():
            return {}
            
        try:
            import json
            with open(config_path) as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Failed to load plugin config: {e}")
            return {}
            
    def _get_plugin_priority(self, manifest_path: Path) -> int:
        """
        获取插件优先级
        :param manifest_path: 清单文件路径
        :return: 优先级值
        """
        try:
            import json
            with open(manifest_path) as f:
                manifest = json.load(f)
                
            priority = manifest.get("priority", "NORMAL").upper()
            return PluginPriority[priority].value
            
        except Exception:
            return PluginPriority.NORMAL.value
            
    def _find_plugin_class(self, module: Any) -> Type[Plugin]:
        """
        查找插件类
        :param module: 模块对象
        :return: 插件类
        """
        for _, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and
                issubclass(obj, Plugin) and
                obj != Plugin):
                return obj
                
        raise ValueError("Plugin class not found")

def create_plugin_manager(
    plugin_dir: Union[str, Path],
    **kwargs
) -> PluginManager:
    """
    创建插件管理器
    :param plugin_dir: 插件目录
    :param kwargs: 附加参数
    :return: 插件管理器实例
    """
    return PluginManager(plugin_dir, **kwargs)
