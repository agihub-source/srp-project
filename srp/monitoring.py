"""监控模块"""

import time
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """指标类型"""
    COUNTER = auto()  # 计数器
    GAUGE = auto()    # 度量值
    HISTOGRAM = auto()  # 直方图
    SUMMARY = auto()   # 摘要统计

@dataclass
class Sample:
    """采样数据"""
    name: str            # 指标名称
    value: float        # 指标值
    timestamp: float    # 采样时间
    labels: Dict        # 标签
    metadata: Dict      # 元数据

@dataclass
class Bucket:
    """直方图桶"""
    le: float           # 上限
    count: int = 0      # 计数

@dataclass
class MetricMetadata:
    """指标元数据"""
    name: str           # 指标名称
    description: str    # 描述
    type: MetricType   # 指标类型
    unit: str = ""     # 单位

class Metric(ABC):
    """指标基类"""
    
    def __init__(
        self,
        name: str,
        description: str,
        label_names: List[str]
    ):
        self.name = name
        self.description = description
        self.label_names = label_names
        self.metadata = MetricMetadata(
            name=name,
            description=description,
            type=self._get_type()
        )
        
    @abstractmethod
    def _get_type(self) -> MetricType:
        """获取指标类型"""
        pass
        
    def _validate_labels(self, labels: Dict[str, str]) -> None:
        """
        验证标签
        :param labels: 标签字典
        """
        if set(labels.keys()) != set(self.label_names):
            raise ValueError(
                f"Invalid labels. Expected: {self.label_names}, got: {list(labels.keys())}"
            )

class Counter(Metric):
    """计数器"""
    
    def __init__(
        self,
        name: str,
        description: str,
        label_names: List[str]
    ):
        super().__init__(name, description, label_names)
        self._values: Dict[str, float] = {}
        self._lock = threading.Lock()
        
    def _get_type(self) -> MetricType:
        return MetricType.COUNTER
        
    def inc(self, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """
        增加计数
        :param value: 增加值
        :param labels: 标签
        """
        if value < 0:
            raise ValueError("Counter value cannot be negative")
            
        labels = labels or {}
        self._validate_labels(labels)
        
        key = self._get_key(labels)
        with self._lock:
            if key not in self._values:
                self._values[key] = 0
            self._values[key] += value
            
    def get(self, labels: Dict[str, str] = None) -> float:
        """
        获取当前值
        :param labels: 标签
        :return: 当前值
        """
        labels = labels or {}
        self._validate_labels(labels)
        
        key = self._get_key(labels)
        with self._lock:
            return self._values.get(key, 0)
            
    def _get_key(self, labels: Dict[str, str]) -> str:
        """
        获取标签键
        :param labels: 标签
        :return: 标签键
        """
        sorted_labels = sorted(labels.items())
        return ",".join(f"{k}={v}" for k, v in sorted_labels)

class Gauge(Metric):
    """度量值"""
    
    def __init__(
        self,
        name: str,
        description: str,
        label_names: List[str]
    ):
        super().__init__(name, description, label_names)
        self._values: Dict[str, float] = {}
        self._lock = threading.Lock()
        
    def _get_type(self) -> MetricType:
        return MetricType.GAUGE
        
    def set(self, value: float, labels: Dict[str, str] = None) -> None:
        """
        设置值
        :param value: 设置值
        :param labels: 标签
        """
        labels = labels or {}
        self._validate_labels(labels)
        
        key = self._get_key(labels)
        with self._lock:
            self._values[key] = value
            
    def inc(self, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """
        增加值
        :param value: 增加值
        :param labels: 标签
        """
        labels = labels or {}
        self._validate_labels(labels)
        
        key = self._get_key(labels)
        with self._lock:
            if key not in self._values:
                self._values[key] = 0
            self._values[key] += value
            
    def dec(self, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """
        减少值
        :param value: 减少值
        :param labels: 标签
        """
        self.inc(-value, labels)
        
    def get(self, labels: Dict[str, str] = None) -> float:
        """
        获取当前值
        :param labels: 标签
        :return: 当前值
        """
        labels = labels or {}
        self._validate_labels(labels)
        
        key = self._get_key(labels)
        with self._lock:
            return self._values.get(key, 0)
            
    def _get_key(self, labels: Dict[str, str]) -> str:
        """
        获取标签键
        :param labels: 标签
        :return: 标签键
        """
        sorted_labels = sorted(labels.items())
        return ",".join(f"{k}={v}" for k, v in sorted_labels)

class Histogram(Metric):
    """直方图"""
    
    def __init__(
        self,
        name: str,
        description: str,
        label_names: List[str],
        buckets: List[float] = None
    ):
        super().__init__(name, description, label_names)
        self.buckets = sorted(buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10])
        self._data: Dict[str, List[Bucket]] = {}
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        
    def _get_type(self) -> MetricType:
        return MetricType.HISTOGRAM
        
    def observe(self, value: float, labels: Dict[str, str] = None) -> None:
        """
        记录观察值
        :param value: 观察值
        :param labels: 标签
        """
        labels = labels or {}
        self._validate_labels(labels)
        
        key = self._get_key(labels)
        with self._lock:
            if key not in self._data:
                self._init_data(key)
                
            # 更新桶
            for bucket in self._data[key]:
                if value <= bucket.le:
                    bucket.count += 1
                    
            # 更新总和和计数
            self._sums[key] += value
            self._counts[key] += 1
            
    def get_buckets(self, labels: Dict[str, str] = None) -> List[Bucket]:
        """
        获取桶数据
        :param labels: 标签
        :return: 桶列表
        """
        labels = labels or {}
        self._validate_labels(labels)
        
        key = self._get_key(labels)
        with self._lock:
            if key not in self._data:
                self._init_data(key)
            return self._data[key].copy()
            
    def get_sum(self, labels: Dict[str, str] = None) -> float:
        """
        获取总和
        :param labels: 标签
        :return: 总和
        """
        labels = labels or {}
        self._validate_labels(labels)
        
        key = self._get_key(labels)
        with self._lock:
            if key not in self._sums:
                self._init_data(key)
            return self._sums[key]
            
    def get_count(self, labels: Dict[str, str] = None) -> int:
        """
        获取计数
        :param labels: 标签
        :return: 计数
        """
        labels = labels or {}
        self._validate_labels(labels)
        
        key = self._get_key(labels)
        with self._lock:
            if key not in self._counts:
                self._init_data(key)
            return self._counts[key]
            
    def _init_data(self, key: str) -> None:
        """
        初始化数据
        :param key: 标签键
        """
        self._data[key] = [Bucket(le=le) for le in self.buckets]
        self._sums[key] = 0
        self._counts[key] = 0
        
    def _get_key(self, labels: Dict[str, str]) -> str:
        """
        获取标签键
        :param labels: 标签
        :return: 标签键
        """
        sorted_labels = sorted(labels.items())
        return ",".join(f"{k}={v}" for k, v in sorted_labels)

@dataclass
class Quantile:
    """分位数"""
    quantile: float     # 分位数
    error: float       # 误差
    value: float = 0.0  # 值

class Summary(Metric):
    """摘要统计"""
    
    def __init__(
        self,
        name: str,
        description: str,
        label_names: List[str],
        quantiles: List[Quantile] = None
    ):
        super().__init__(name, description, label_names)
        self.quantiles = quantiles or [
            Quantile(0.5, 0.05),   # 中位数
            Quantile(0.9, 0.01),   # P90
            Quantile(0.99, 0.001)  # P99
        ]
        self._values: Dict[str, List[float]] = {}
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        
    def _get_type(self) -> MetricType:
        return MetricType.SUMMARY
        
    def observe(self, value: float, labels: Dict[str, str] = None) -> None:
        """
        记录观察值
        :param value: 观察值
        :param labels: 标签
        """
        labels = labels or {}
        self._validate_labels(labels)
        
        key = self._get_key(labels)
        with self._lock:
            if key not in self._values:
                self._init_data(key)
                
            self._values[key].append(value)
            self._sums[key] += value
            self._counts[key] += 1
            
    def get_quantiles(self, labels: Dict[str, str] = None) -> List[Quantile]:
        """
        获取分位数
        :param labels: 标签
        :return: 分位数列表
        """
        labels = labels or {}
        self._validate_labels(labels)
        
        key = self._get_key(labels)
        with self._lock:
            if key not in self._values:
                self._init_data(key)
                return self.quantiles.copy()
                
            values = sorted(self._values[key])
            if not values:
                return self.quantiles.copy()
                
            result = []
            for q in self.quantiles:
                idx = int(q.quantile * len(values))
                q.value = values[idx]
                result.append(q)
                
            return result
            
    def get_sum(self, labels: Dict[str, str] = None) -> float:
        """
        获取总和
        :param labels: 标签
        :return: 总和
        """
        labels = labels or {}
        self._validate_labels(labels)
        
        key = self._get_key(labels)
        with self._lock:
            if key not in self._sums:
                self._init_data(key)
            return self._sums[key]
            
    def get_count(self, labels: Dict[str, str] = None) -> int:
        """
        获取计数
        :param labels: 标签
        :return: 计数
        """
        labels = labels or {}
        self._validate_labels(labels)
        
        key = self._get_key(labels)
        with self._lock:
            if key not in self._counts:
                self._init_data(key)
            return self._counts[key]
            
    def _init_data(self, key: str) -> None:
        """
        初始化数据
        :param key: 标签键
        """
        self._values[key] = []
        self._sums[key] = 0
        self._counts[key] = 0
        
    def _get_key(self, labels: Dict[str, str]) -> str:
        """
        获取标签键
        :param labels: 标签
        :return: 标签键
        """
        sorted_labels = sorted(labels.items())
        return ",".join(f"{k}={v}" for k, v in sorted_labels)

class MetricFamily:
    """指标族"""
    
    def __init__(
        self,
        name: str,
        description: str,
        type: MetricType,
        label_names: List[str]
    ):
        self.name = name
        self.description = description
        self.type = type
        self.label_names = label_names
        
    def create_metric(self, **kwargs) -> Metric:
        """
        创建指标
        :param kwargs: 附加参数
        :return: 指标实例
        """
        if self.type == MetricType.COUNTER:
            return Counter(self.name, self.description, self.label_names)
        elif self.type == MetricType.GAUGE:
            return Gauge(self.name, self.description, self.label_names)
        elif self.type == MetricType.HISTOGRAM:
            return Histogram(self.name, self.description, self.label_names, **kwargs)
        elif self.type == MetricType.SUMMARY:
            return Summary(self.name, self.description, self.label_names, **kwargs)
        else:
            raise ValueError(f"Unknown metric type: {self.type}")

class MetricRegistry:
    """指标注册器"""
    
    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()
        
    def register(self, metric: Metric) -> None:
        """
        注册指标
        :param metric: 指标
        """
        with self._lock:
            if metric.name in self._metrics:
                raise ValueError(f"Metric already exists: {metric.name}")
            self._metrics[metric.name] = metric
            
    def unregister(self, name: str) -> None:
        """
        注销指标
        :param name: 指标名称
        """
        with self._lock:
            if name in self._metrics:
                del self._metrics[name]
                
    def get_metric(self, name: str) -> Optional[Metric]:
        """
        获取指标
        :param name: 指标名称
        :return: 指标实例
        """
        return self._metrics.get(name)
        
    def get_metrics(self) -> List[Metric]:
        """
        获取所有指标
        :return: 指标列表
        """
        return list(self._metrics.values())

class Monitoring:
    """监控管理器"""
    
    def __init__(self):
        self.registry = MetricRegistry()
        
    def counter(
        self,
        name: str,
        description: str,
        label_names: List[str] = None
    ) -> Counter:
        """
        创建计数器
        :param name: 指标名称
        :param description: 描述
        :param label_names: 标签名列表
        :return: 计数器实例
        """
        label_names = label_names or []
        counter = Counter(name, description, label_names)
        self.registry.register(counter)
        return counter
        
    def gauge(
        self,
        name: str,
        description: str,
        label_names: List[str] = None
    ) -> Gauge:
        """
        创建度量值
        :param name: 指标名称
        :param description: 描述
        :param label_names: 标签名列表
        :return: 度量值实例
        """
        label_names = label_names or []
        gauge = Gauge(name, description, label_names)
        self.registry.register(gauge)
        return gauge
        
    def histogram(
        self,
        name: str,
        description: str,
        label_names: List[str] = None,
        buckets: List[float] = None
    ) -> Histogram:
        """
        创建直方图
        :param name: 指标名称
        :param description: 描述
        :param label_names: 标签名列表
        :param buckets: 桶上限列表
        :return: 直方图实例
        """
        label_names = label_names or []
        histogram = Histogram(name, description, label_names, buckets)
        self.registry.register(histogram)
        return histogram
        
    def summary(
        self,
        name: str,
        description: str,
        label_names: List[str] = None,
        quantiles: List[Quantile] = None
    ) -> Summary:
        """
        创建摘要统计
        :param name: 指标名称
        :param description: 描述
        :param label_names: 标签名列表
        :param quantiles: 分位数列表
        :return: 摘要统计实例
        """
        label_names = label_names or []
        summary = Summary(name, description, label_names, quantiles)
        self.registry.register(summary)
        return summary
        
    @contextmanager
    def timer(
        self,
        name: str,
        description: str = "",
        label_names: List[str] = None,
        labels: Dict[str, str] = None
    ):
        """
        计时器上下文管理器
        :param name: 指标名称
        :param description: 描述
        :param label_names: 标签名列表
        :param labels: 标签值字典
        """
        label_names = label_names or []
        labels = labels or {}
        
        # 确保存在直方图
        histogram = self.registry.get_metric(name)
        if not histogram:
            histogram = self.histogram(
                name,
                description or f"Timer histogram for {name}",
                label_names
            )
            
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            histogram.observe(duration, labels)

# 创建全局监控实例
monitoring = Monitoring()
