"""安全功能示例"""

import asyncio
import logging
from pathlib import Path

from srp.security.encryption import (
    EncryptionAlgorithm,
    HashAlgorithm,
    SecurityManager,
    create_security_manager
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    try:
        # 创建安全管理器
        security = create_security_manager(
            default_algorithm=EncryptionAlgorithm.AES,
            key_size=2048,
            key_rotation_days=30
        )
        
        # 演示RSA加密
        logger.info("=== RSA Encryption Demo ===")
        rsa_key_pair = security.generate_key_pair(EncryptionAlgorithm.RSA)
        message = "This is a secret message"
        
        # 加密
        encrypted = security.encrypt(
            message,
            rsa_key_pair,
            EncryptionAlgorithm.RSA
        )
        logger.info(f"Encrypted (RSA): {encrypted.ciphertext.hex()}")
        
        # 解密
        decrypted = security.decrypt(encrypted, rsa_key_pair)
        logger.info(f"Decrypted (RSA): {decrypted.decode()}")
        
        # 演示AES加密
        logger.info("\n=== AES Encryption Demo ===")
        aes_key_pair = security.generate_key_pair(EncryptionAlgorithm.AES)
        
        # 加密
        encrypted = security.encrypt(
            message,
            aes_key_pair,
            EncryptionAlgorithm.AES
        )
        logger.info(f"Encrypted (AES): {encrypted.ciphertext.hex()}")
        logger.info(f"IV: {encrypted.iv.hex()}")
        
        # 解密
        decrypted = security.decrypt(encrypted, aes_key_pair)
        logger.info(f"Decrypted (AES): {decrypted.decode()}")
        
        # 演示SM4加密(国密算法)
        logger.info("\n=== SM4 Encryption Demo ===")
        sm4_key_pair = security.generate_key_pair(EncryptionAlgorithm.SM4)
        
        # 加密
        encrypted = security.encrypt(
            message,
            sm4_key_pair,
            EncryptionAlgorithm.SM4
        )
        logger.info(f"Encrypted (SM4): {encrypted.ciphertext.hex()}")
        logger.info(f"IV: {encrypted.iv.hex()}")
        
        # 解密
        decrypted = security.decrypt(encrypted, sm4_key_pair)
        logger.info(f"Decrypted (SM4): {decrypted.decode()}")
        
        # 演示哈希计算
        logger.info("\n=== Hash Computation Demo ===")
        data = b"Hello, World!"
        
        # SHA-256
        hash_value = security.compute_hash(data, HashAlgorithm.SHA256)
        logger.info(f"SHA-256: {hash_value.hex()}")
        
        # SHA-384
        hash_value = security.compute_hash(data, HashAlgorithm.SHA384)
        logger.info(f"SHA-384: {hash_value.hex()}")
        
        # SHA-512
        hash_value = security.compute_hash(data, HashAlgorithm.SHA512)
        logger.info(f"SHA-512: {hash_value.hex()}")
        
        # BLAKE2b
        hash_value = security.compute_hash(data, HashAlgorithm.BLAKE2B)
        logger.info(f"BLAKE2b: {hash_value.hex()}")
        
        # SM3(国密算法)
        try:
            hash_value = security.compute_hash(data, HashAlgorithm.SM3)
            logger.info(f"SM3: {hash_value.hex()}")
        except ImportError:
            logger.warning("SM3 hash requires gmssl package")
            
        # 演示密钥轮换
        logger.info("\n=== Key Rotation Demo ===")
        # 创建一些活跃密钥
        for i in range(3):
            key_pair = security.generate_key_pair()
            logger.info(f"Generated key pair {i + 1}")
            
        # 执行密钥轮换
        await asyncio.sleep(2)  # 模拟时间流逝
        logger.info("Performing key rotation...")
        security.rotate_keys()
        logger.info("Key rotation completed")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
        raise

if __name__ == "__main__":
    # Windows需要使用这种方式运行asyncio
    asyncio.run(main())
