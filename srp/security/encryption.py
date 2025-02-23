"""加密模块"""

import base64
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.asymmetric import padding as asymmetric_padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
    load_pem_private_key,
    load_pem_public_key,
)

from ..monitoring import monitoring

logger = logging.getLogger(__name__)

class EncryptionAlgorithm(Enum):
    """加密算法"""
    AES = auto()       # AES对称加密
    RSA = auto()       # RSA非对称加密
    FERNET = auto()    # Fernet对称加密
    SM4 = auto()       # 国密SM4分组密码
    CHACHA20 = auto()  # ChaCha20流密码

class HashAlgorithm(Enum):
    """哈希算法"""
    SHA256 = auto()    # SHA-256
    SHA384 = auto()    # SHA-384
    SHA512 = auto()    # SHA-512
    SM3 = auto()       # 国密SM3杂凑算法
    BLAKE2B = auto()   # BLAKE2b

@dataclass
class KeyPair:
    """密钥对"""
    private_key: bytes
    public_key: bytes
    algorithm: EncryptionAlgorithm
    created_at: float
    expires_at: Optional[float] = None
    metadata: Dict = None

@dataclass
class EncryptionResult:
    """加密结果"""
    ciphertext: bytes
    iv: Optional[bytes] = None
    tag: Optional[bytes] = None
    algorithm: Optional[EncryptionAlgorithm] = None
    metadata: Dict = None

class SecurityManager:
    """安全管理器"""
    
    def __init__(
        self,
        default_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES,
        key_size: int = 2048,
        key_rotation_days: int = 30
    ):
        self.default_algorithm = default_algorithm
        self.key_size = key_size
        self.key_rotation_days = key_rotation_days
        
        # 密钥管理
        self._active_keys: Dict[str, KeyPair] = {}
        self._revoked_keys: Dict[str, KeyPair] = {}
        
        # 监控指标
        self.encryption_counter = monitoring.counter(
            "srp_encryption_operations_total",
            "Total number of encryption operations",
            ["algorithm", "operation"]
        )
        
        self.key_gauge = monitoring.gauge(
            "srp_active_keys",
            "Number of active encryption keys",
            ["algorithm"]
        )
        
        self.operation_histogram = monitoring.histogram(
            "srp_encryption_operation_seconds",
            "Encryption operation duration in seconds",
            ["algorithm", "operation"]
        )
        
        self.error_counter = monitoring.counter(
            "srp_encryption_errors_total",
            "Total number of encryption errors",
            ["algorithm", "error_type"]
        )
        
    def generate_key_pair(
        self,
        algorithm: Optional[EncryptionAlgorithm] = None
    ) -> KeyPair:
        """
        生成密钥对
        :param algorithm: 加密算法
        :return: 密钥对
        """
        algorithm = algorithm or self.default_algorithm
        
        try:
            if algorithm == EncryptionAlgorithm.RSA:
                # 生成RSA密钥对
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=self.key_size
                )
                
                public_key = private_key.public_key()
                
                # 序列化密钥
                private_pem = private_key.private_bytes(
                    encoding=Encoding.PEM,
                    format=PrivateFormat.PKCS8,
                    encryption_algorithm=NoEncryption()
                )
                
                public_pem = public_key.public_bytes(
                    encoding=Encoding.PEM,
                    format=PublicFormat.SubjectPublicKeyInfo
                )
                
            elif algorithm in [EncryptionAlgorithm.AES, EncryptionAlgorithm.SM4]:
                # 生成对称密钥
                key = os.urandom(32)  # 256位密钥
                private_pem = key
                public_pem = key
                
            elif algorithm == EncryptionAlgorithm.FERNET:
                # 生成Fernet密钥
                key = Fernet.generate_key()
                private_pem = key
                public_pem = key
                
            elif algorithm == EncryptionAlgorithm.CHACHA20:
                # 生成ChaCha20密钥
                key = os.urandom(32)
                private_pem = key
                public_pem = key
                
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
            # 创建密钥对
            import time
            key_pair = KeyPair(
                private_key=private_pem,
                public_key=public_pem,
                algorithm=algorithm,
                created_at=time.time(),
                expires_at=time.time() + self.key_rotation_days * 24 * 3600,
                metadata={}
            )
            
            # 更新监控指标
            self.key_gauge.inc(
                labels={"algorithm": algorithm.name.lower()}
            )
            
            return key_pair
            
        except Exception as e:
            logger.error(f"Failed to generate key pair: {e}")
            self.error_counter.inc(
                labels={
                    "algorithm": algorithm.name.lower(),
                    "error_type": "key_generation"
                }
            )
            raise
            
    def encrypt(
        self,
        data: Union[str, bytes],
        key: Union[bytes, KeyPair],
        algorithm: Optional[EncryptionAlgorithm] = None
    ) -> EncryptionResult:
        """
        加密数据
        :param data: 待加密数据
        :param key: 加密密钥
        :param algorithm: 加密算法
        :return: 加密结果
        """
        import time
        start_time = time.time()
        
        try:
            # 处理输入数据
            if isinstance(data, str):
                data = data.encode()
                
            if isinstance(key, KeyPair):
                key = key.public_key
                algorithm = algorithm or key.algorithm
                
            algorithm = algorithm or self.default_algorithm
            
            if algorithm == EncryptionAlgorithm.RSA:
                # RSA加密
                public_key = load_pem_public_key(key)
                ciphertext = public_key.encrypt(
                    data,
                    asymmetric_padding.OAEP(
                        mgf=asymmetric_padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                result = EncryptionResult(
                    ciphertext=ciphertext,
                    algorithm=algorithm
                )
                
            elif algorithm in [EncryptionAlgorithm.AES, EncryptionAlgorithm.SM4]:
                # AES/SM4加密
                iv = os.urandom(16)
                cipher = Cipher(
                    algorithms.AES(key) if algorithm == EncryptionAlgorithm.AES
                    else algorithms.SM4(key),
                    modes.CBC(iv)
                )
                
                encryptor = cipher.encryptor()
                padder = padding.PKCS7(128).padder()
                
                padded_data = padder.update(data) + padder.finalize()
                ciphertext = encryptor.update(padded_data) + encryptor.finalize()
                
                result = EncryptionResult(
                    ciphertext=ciphertext,
                    iv=iv,
                    algorithm=algorithm
                )
                
            elif algorithm == EncryptionAlgorithm.FERNET:
                # Fernet加密
                f = Fernet(key)
                ciphertext = f.encrypt(data)
                result = EncryptionResult(
                    ciphertext=ciphertext,
                    algorithm=algorithm
                )
                
            elif algorithm == EncryptionAlgorithm.CHACHA20:
                # ChaCha20加密
                nonce = os.urandom(16)
                algorithm = algorithms.ChaCha20(key, nonce)
                cipher = Cipher(algorithm, mode=None)
                
                encryptor = cipher.encryptor()
                ciphertext = encryptor.update(data)
                
                result = EncryptionResult(
                    ciphertext=ciphertext,
                    iv=nonce,
                    algorithm=algorithm
                )
                
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
            # 更新监控指标
            duration = time.time() - start_time
            
            self.encryption_counter.inc(
                labels={
                    "algorithm": algorithm.name.lower(),
                    "operation": "encrypt"
                }
            )
            
            self.operation_histogram.observe(
                duration,
                labels={
                    "algorithm": algorithm.name.lower(),
                    "operation": "encrypt"
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            self.error_counter.inc(
                labels={
                    "algorithm": algorithm.name.lower() if algorithm else "unknown",
                    "error_type": "encryption"
                }
            )
            raise
            
    def decrypt(
        self,
        encrypted_data: EncryptionResult,
        key: Union[bytes, KeyPair]
    ) -> bytes:
        """
        解密数据
        :param encrypted_data: 加密结果
        :param key: 解密密钥
        :return: 解密后的数据
        """
        import time
        start_time = time.time()
        
        try:
            if isinstance(key, KeyPair):
                key = key.private_key
                
            algorithm = encrypted_data.algorithm
            
            if algorithm == EncryptionAlgorithm.RSA:
                # RSA解密
                private_key = load_pem_private_key(key, password=None)
                plaintext = private_key.decrypt(
                    encrypted_data.ciphertext,
                    asymmetric_padding.OAEP(
                        mgf=asymmetric_padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                
            elif algorithm in [EncryptionAlgorithm.AES, EncryptionAlgorithm.SM4]:
                # AES/SM4解密
                cipher = Cipher(
                    algorithms.AES(key) if algorithm == EncryptionAlgorithm.AES
                    else algorithms.SM4(key),
                    modes.CBC(encrypted_data.iv)
                )
                
                decryptor = cipher.decryptor()
                unpadder = padding.PKCS7(128).unpadder()
                
                padded_plaintext = decryptor.update(
                    encrypted_data.ciphertext
                ) + decryptor.finalize()
                plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
                
            elif algorithm == EncryptionAlgorithm.FERNET:
                # Fernet解密
                f = Fernet(key)
                plaintext = f.decrypt(encrypted_data.ciphertext)
                
            elif algorithm == EncryptionAlgorithm.CHACHA20:
                # ChaCha20解密
                algorithm = algorithms.ChaCha20(key, encrypted_data.iv)
                cipher = Cipher(algorithm, mode=None)
                
                decryptor = cipher.decryptor()
                plaintext = decryptor.update(encrypted_data.ciphertext)
                
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
            # 更新监控指标
            duration = time.time() - start_time
            
            self.encryption_counter.inc(
                labels={
                    "algorithm": algorithm.name.lower(),
                    "operation": "decrypt"
                }
            )
            
            self.operation_histogram.observe(
                duration,
                labels={
                    "algorithm": algorithm.name.lower(),
                    "operation": "decrypt"
                }
            )
            
            return plaintext
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            self.error_counter.inc(
                labels={
                    "algorithm": algorithm.name.lower() if algorithm else "unknown",
                    "error_type": "decryption"
                }
            )
            raise
            
    def compute_hash(
        self,
        data: Union[str, bytes],
        algorithm: HashAlgorithm = HashAlgorithm.SHA256
    ) -> bytes:
        """
        计算哈希值
        :param data: 数据
        :param algorithm: 哈希算法
        :return: 哈希值
        """
        try:
            if isinstance(data, str):
                data = data.encode()
                
            if algorithm == HashAlgorithm.SHA256:
                return hashlib.sha256(data).digest()
            elif algorithm == HashAlgorithm.SHA384:
                return hashlib.sha384(data).digest()
            elif algorithm == HashAlgorithm.SHA512:
                return hashlib.sha512(data).digest()
            elif algorithm == HashAlgorithm.BLAKE2B:
                return hashlib.blake2b(data).digest()
            elif algorithm == HashAlgorithm.SM3:
                # 需要额外安装gmssl库
                from gmssl import sm3
                return bytes.fromhex(sm3.sm3_hash(data))
            else:
                raise ValueError(f"Unsupported hash algorithm: {algorithm}")
                
        except Exception as e:
            logger.error(f"Failed to compute hash: {e}")
            self.error_counter.inc(
                labels={
                    "algorithm": algorithm.name.lower(),
                    "error_type": "hash"
                }
            )
            raise
            
    def rotate_keys(self) -> None:
        """密钥轮换"""
        import time
        current_time = time.time()
        
        try:
            # 检查过期密钥
            expired_keys = []
            for key_id, key_pair in self._active_keys.items():
                if (key_pair.expires_at and
                    current_time > key_pair.expires_at):
                    expired_keys.append(key_id)
                    
            # 移动过期密钥
            for key_id in expired_keys:
                key_pair = self._active_keys.pop(key_id)
                self._revoked_keys[key_id] = key_pair
                
                # 更新监控指标
                self.key_gauge.dec(
                    labels={"algorithm": key_pair.algorithm.name.lower()}
                )
                
            # 生成新密钥
            for key_id in expired_keys:
                old_key = self._revoked_keys[key_id]
                new_key = self.generate_key_pair(old_key.algorithm)
                self._active_keys[key_id] = new_key
                
            logger.info(f"Rotated {len(expired_keys)} keys")
            
        except Exception as e:
            logger.error(f"Failed to rotate keys: {e}")
            self.error_counter.inc(
                labels={
                    "algorithm": "all",
                    "error_type": "key_rotation"
                }
            )
            raise

def create_security_manager(**kwargs) -> SecurityManager:
    """
    创建安全管理器
    :param kwargs: 附加参数
    :return: 安全管理器实例
    """
    return SecurityManager(**kwargs)
