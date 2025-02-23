"""
合规性检查模块 - 实现数据合规性和AI伦理检查
包括数据本地化、隐私保护、AI伦理等方面的检查
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import re
import json
import datetime

class ComplianceError(Exception):
    """合规性错误"""
    pass

class DataLocationType(Enum):
    """数据位置类型"""
    CHINA_MAINLAND = "China"
    HONG_KONG = "Hong Kong"
    OVERSEAS = "Overseas"

class DataSensitivityLevel(Enum):
    """数据敏感度级别"""
    PUBLIC = 1      # 公开数据
    INTERNAL = 2    # 内部数据
    SENSITIVE = 3   # 敏感数据
    CLASSIFIED = 4  # 机密数据

class AIEthicsLevel(Enum):
    """AI伦理级别"""
    LOW_RISK = 1    # 低风险
    MEDIUM_RISK = 2 # 中等风险
    HIGH_RISK = 3   # 高风险
    FORBIDDEN = 4   # 禁止使用

class ComplianceChecker:
    """合规性检查器"""
    
    def __init__(self):
        """初始化合规性检查器"""
        self.sensitive_patterns = [
            r'\d{18}',  # 身份证号
            r'\d{11}',  # 手机号
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # 邮箱
        ]
        
        self.forbidden_words = [
            '色情', '赌博', '毒品', '暴力'
        ]

    def check_data_compliance(self, data: Dict[str, Any]) -> bool:
        """
        检查数据合规性
        :param data: 要检查的数据
        :return: 是否合规
        """
        self._check_data_location(data)
        self._check_privacy_compliance(data)
        self._check_ai_ethics(data)
        return True

    def _check_data_location(self, data: Dict[str, Any]):
        """
        检查数据本地化要求
        :param data: 要检查的数据
        """
        location = data.get('location')
        if not location:
            raise ComplianceError("未指定数据位置")
        
        if location != DataLocationType.CHINA_MAINLAND.value:
            raise ComplianceError("数据必须存储在中国大陆地区")
        
        # 检查是否有跨境传输需求
        if data.get('cross_border_transfer'):
            if not data.get('cross_border_permission'):
                raise ComplianceError("跨境数据传输需要相关许可")

    def _check_privacy_compliance(self, data: Dict[str, Any]):
        """
        检查隐私合规性
        :param data: 要检查的数据
        """
        # 检查是否包含个人信息
        if data.get('personal_info'):
            if not data.get('consent'):
                raise ComplianceError("处理个人信息需要用户同意")
            
            if not data.get('privacy_policy'):
                raise ComplianceError("缺少隐私政策声明")
        
        # 检查敏感信息
        content = json.dumps(data)
        for pattern in self.sensitive_patterns:
            if re.search(pattern, content):
                if not data.get('encryption'):
                    raise ComplianceError("敏感信息必须加密存储")

    def _check_ai_ethics(self, data: Dict[str, Any]):
        """
        检查AI伦理合规性
        :param data: 要检查的数据
        """
        # 检查AI模型用途
        if 'ai_purpose' in data:
            purpose = data['ai_purpose']
            if any(word in purpose for word in self.forbidden_words):
                raise ComplianceError("AI用途违反伦理要求")
        
        # 检查模型偏差
        if 'bias' in data:
            bias = float(data['bias'])
            if bias > 0.5:  # 偏差阈值
                raise ComplianceError("模型偏差超出允许范围")
        
        # 检查决策透明度
        if data.get('automated_decision'):
            if not data.get('explanation_method'):
                raise ComplianceError("自动化决策需要提供解释方法")

class DataComplianceValidator:
    """数据合规性验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.checker = ComplianceChecker()
        self._compliance_records = []

    def validate(self, data: Dict[str, Any]) -> bool:
        """
        验证数据合规性
        :param data: 要验证的数据
        :return: 是否合规
        """
        try:
            result = self.checker.check_data_compliance(data)
            self._record_check(data, True)
            return result
        except ComplianceError as e:
            self._record_check(data, False, str(e))
            raise

    def _record_check(self, data: Dict[str, Any], passed: bool, error: str = None):
        """
        记录合规性检查结果
        :param data: 检查的数据
        :param passed: 是否通过
        :param error: 错误信息
        """
        record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'data_type': type(data).__name__,
            'sensitivity_level': self._get_sensitivity_level(data),
            'passed': passed,
            'error': error
        }
        self._compliance_records.append(record)

    def _get_sensitivity_level(self, data: Dict[str, Any]) -> DataSensitivityLevel:
        """
        获取数据敏感度级别
        :param data: 数据
        :return: 敏感度级别
        """
        if data.get('classified'):
            return DataSensitivityLevel.CLASSIFIED
        elif data.get('personal_info'):
            return DataSensitivityLevel.SENSITIVE
        elif data.get('internal_only'):
            return DataSensitivityLevel.INTERNAL
        return DataSensitivityLevel.PUBLIC

    def get_compliance_report(self) -> List[Dict[str, Any]]:
        """
        获取合规性检查报告
        :return: 检查记录列表
        """
        return self._compliance_records

def check_data_compliance(data: Dict[str, Any]) -> bool:
    """
    便捷的合规性检查函数
    :param data: 要检查的数据
    :return: 是否合规
    """
    validator = DataComplianceValidator()
    return validator.validate(data)

# 使用示例
if __name__ == "__main__":
    # 创建验证器
    validator = DataComplianceValidator()
    
    # 测试合规数据
    compliant_data = {
        "location": "China",
        "personal_info": True,
        "consent": True,
        "privacy_policy": True,
        "encryption": True,
        "ai_purpose": "数据分析",
        "bias": 0.3,
        "automated_decision": True,
        "explanation_method": "LIME"
    }
    
    try:
        validator.validate(compliant_data)
        print("数据合规")
    except ComplianceError as e:
        print(f"合规性错误: {e}")
    
    # 测试不合规数据
    non_compliant_data = {
        "location": "USA",
        "personal_info": True,
        "consent": False
    }
    
    try:
        validator.validate(non_compliant_data)
    except ComplianceError as e:
        print(f"预期的合规性错误: {e}")
    
    # 获取合规性报告
    report = validator.get_compliance_report()
    print("\n合规性检查报告:")
    for record in report:
        print(json.dumps(record, indent=2, ensure_ascii=False))