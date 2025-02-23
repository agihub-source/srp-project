"""会话管理模块"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set

from ..monitoring import monitoring

logger = logging.getLogger(__name__)

class SessionState(Enum):
    """会话状态"""
    INIT = auto()       # 初始化
    ACTIVE = auto()     # 活跃
    IDLE = auto()       # 空闲
    CLOSED = auto()     # 关闭
    ERROR = auto()      # 错误

@dataclass
class SessionMetrics:
    """会话指标"""
    messages_sent: int = 0      # 发送消息数
    messages_received: int = 0   # 接收消息数
    bytes_sent: int = 0         # 发送字节数
    bytes_received: int = 0     # 接收字节数
    errors: int = 0             # 错误数

@dataclass
class Session:
    """会话"""
    session_id: str              # 会话ID
    peer_id: str                # 对端ID
    created_at: float           # 创建时间
    expires_at: float           # 过期时间
    state: SessionState = SessionState.INIT  # 会话状态
    metadata: Dict = field(default_factory=dict)  # 元数据
    metrics: SessionMetrics = field(default_factory=SessionMetrics)  # 会话指标

class SessionManager:
    """会话管理器"""
    
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._peers: Dict[str, Set[str]] = {}  # peer_id -> session_ids
        self._lock = threading.Lock()
        
        # 监控指标
        self.session_gauge = monitoring.gauge(
            "srp_active_sessions",
            "Number of active sessions",
            ["state"]
        )
        
        self.message_counter = monitoring.counter(
            "srp_session_messages_total",
            "Total number of session messages",
            ["type", "peer_id"]
        )
        
        self.bytes_counter = monitoring.counter(
            "srp_session_bytes_total",
            "Total number of session bytes",
            ["type", "peer_id"]
        )
        
        self.error_counter = monitoring.counter(
            "srp_session_errors_total",
            "Total number of session errors",
            ["peer_id"]
        )
        
    def create_session(
        self,
        peer_id: str,
        ttl: int = 3600,
        metadata: Dict = None
    ) -> Session:
        """
        创建会话
        :param peer_id: 对端ID
        :param ttl: 生存时间(秒)
        :param metadata: 元数据
        :return: 会话实例
        """
        try:
            now = time.time()
            session = Session(
                session_id=str(uuid.uuid4()),
                peer_id=peer_id,
                created_at=now,
                expires_at=now + ttl,
                metadata=metadata or {}
            )
            
            with self._lock:
                self._sessions[session.session_id] = session
                if peer_id not in self._peers:
                    self._peers[peer_id] = set()
                self._peers[peer_id].add(session.session_id)
                
            self.session_gauge.inc(
                labels={"state": session.state.name.lower()}
            )
            
            logger.info(
                f"Created session {session.session_id} for peer {peer_id}"
            )
            return session
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
            
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        获取会话
        :param session_id: 会话ID
        :return: 会话实例
        """
        try:
            with self._lock:
                session = self._sessions.get(session_id)
                if not session:
                    return None
                    
                # 检查过期
                if time.time() > session.expires_at:
                    self.close_session(session_id)
                    return None
                    
                return session
                
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
            
    def get_sessions_by_peer(self, peer_id: str) -> List[Session]:
        """
        获取对端的所有会话
        :param peer_id: 对端ID
        :return: 会话列表
        """
        try:
            with self._lock:
                if peer_id not in self._peers:
                    return []
                    
                sessions = []
                for session_id in self._peers[peer_id]:
                    if session := self.get_session(session_id):
                        sessions.append(session)
                return sessions
                
        except Exception as e:
            logger.error(f"Failed to get sessions by peer: {e}")
            return []
            
    def update_session(
        self,
        session_id: str,
        state: SessionState = None,
        metadata: Dict = None,
        extend_ttl: int = None
    ) -> Optional[Session]:
        """
        更新会话
        :param session_id: 会话ID
        :param state: 新状态
        :param metadata: 新元数据
        :param extend_ttl: 延长生存时间(秒)
        :return: 更新后的会话实例
        """
        try:
            with self._lock:
                session = self._sessions.get(session_id)
                if not session:
                    return None
                    
                # 更新状态
                if state and state != session.state:
                    old_state = session.state
                    session.state = state
                    self.session_gauge.dec(
                        labels={"state": old_state.name.lower()}
                    )
                    self.session_gauge.inc(
                        labels={"state": state.name.lower()}
                    )
                    
                # 更新元数据
                if metadata:
                    session.metadata.update(metadata)
                    
                # 延长生存时间
                if extend_ttl:
                    session.expires_at = time.time() + extend_ttl
                    
                return session
                
        except Exception as e:
            logger.error(f"Failed to update session: {e}")
            return None
            
    def close_session(self, session_id: str) -> bool:
        """
        关闭会话
        :param session_id: 会话ID
        :return: 是否成功
        """
        try:
            with self._lock:
                session = self._sessions.get(session_id)
                if not session:
                    return False
                    
                # 更新状态指标
                self.session_gauge.dec(
                    labels={"state": session.state.name.lower()}
                )
                
                # 清理会话
                del self._sessions[session_id]
                self._peers[session.peer_id].remove(session_id)
                if not self._peers[session.peer_id]:
                    del self._peers[session.peer_id]
                    
                logger.info(
                    f"Closed session {session_id} for peer {session.peer_id}"
                )
                return True
                
        except Exception as e:
            logger.error(f"Failed to close session: {e}")
            return False
            
    def clear_expired_sessions(self) -> int:
        """
        清理过期会话
        :return: 清理数量 
        """
        try:
            now = time.time()
            expired = []
            
            with self._lock:
                for session_id, session in self._sessions.items():
                    if now > session.expires_at:
                        expired.append(session_id)
                        
                for session_id in expired:
                    self.close_session(session_id)
                    
            return len(expired)
            
        except Exception as e:
            logger.error(f"Failed to clear expired sessions: {e}")
            return 0
            
    def record_metrics(
        self,
        session_id: str,
        sent_messages: int = 0,
        received_messages: int = 0,
        sent_bytes: int = 0,
        received_bytes: int = 0,
        errors: int = 0
    ) -> None:
        """
        记录会话指标
        :param session_id: 会话ID
        :param sent_messages: 发送消息数
        :param received_messages: 接收消息数
        :param sent_bytes: 发送字节数
        :param received_bytes: 接收字节数
        :param errors: 错误数
        """
        try:
            with self._lock:
                session = self._sessions.get(session_id)
                if not session:
                    return
                    
                # 更新会话指标
                if sent_messages:
                    session.metrics.messages_sent += sent_messages
                    self.message_counter.inc(
                        value=sent_messages,
                        labels={
                            "type": "sent",
                            "peer_id": session.peer_id
                        }
                    )
                    
                if received_messages:
                    session.metrics.messages_received += received_messages
                    self.message_counter.inc(
                        value=received_messages,
                        labels={
                            "type": "received",
                            "peer_id": session.peer_id
                        }
                    )
                    
                if sent_bytes:
                    session.metrics.bytes_sent += sent_bytes
                    self.bytes_counter.inc(
                        value=sent_bytes,
                        labels={
                            "type": "sent",
                            "peer_id": session.peer_id
                        }
                    )
                    
                if received_bytes:
                    session.metrics.bytes_received += received_bytes
                    self.bytes_counter.inc(
                        value=received_bytes,
                        labels={
                            "type": "received",
                            "peer_id": session.peer_id
                        }
                    )
                    
                if errors:
                    session.metrics.errors += errors
                    self.error_counter.inc(
                        value=errors,
                        labels={
                            "peer_id": session.peer_id
                        }
                    )
                    
        except Exception as e:
            logger.error(f"Failed to record session metrics: {e}")

# 创建全局会话管理器实例
session_manager: Optional[SessionManager] = None

def get_session_manager() -> SessionManager:
    """
    获取会话管理器实例
    :return: 会话管理器实例
    """
    global session_manager
    if not session_manager:
        session_manager = SessionManager()
    return session_manager

# 初始化全局会话管理器
session_manager = get_session_manager()
