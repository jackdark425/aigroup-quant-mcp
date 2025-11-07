"""
模型训练器
支持多种传统机器学习算法
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
import logging
import os
import pickle

from ..logger import get_logger
from ..config import get_config


class ModelTrainer:
    """模型训练器 - 支持多种传统机器学习算法"""
    
    def __init__(self, model_type: str = "lightgbm", model_id: Optional[str] = None):
        """
        初始化模型训练器
        
        Args:
            model_type: 模型类型，支持以下算法：
                - 线性模型: linear, ridge, lasso, elasticnet, logistic
                - 基于树的模型: decision_tree, random_forest, gradient_boosting, lightgbm, xgboost, catboost
                - 支持向量机: svm, svr
                - 朴素贝叶斯: naive_bayes
                - K-最近邻: knn
            model_id: 模型ID，用于缓存和标识模型
        """
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.model_type = model_type or self.config.get('default_model_type', 'lightgbm')
        self.model_id = model_id or f"{model_type}_{id(self)}"
        self.model = None
        self.feature_importance = None
        self.model_params = {}
        self.cache_dir = ".model_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger.info(f"Initialized ModelTrainer with model_type: {self.model_type}, model_id: {self.model_id}")
    
    def prepare_dataset(
        self,
        factors: pd.DataFrame,
        labels: pd.Series,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str
    ) -> tuple:
        """
        准备训练数据集
        
        Returns:
            (X_train, y_train, X_test, y_test)
        """
        self.logger.debug("Preparing dataset")
        # 训练集
        train_mask = (
            (factors.index.get_level_values(0) >= train_start) &
            (factors.index.get_level_values(0) <= train_end)
        )
        X_train = factors[train_mask]
        y_train = labels[train_mask]
        
        # 测试集
        test_mask = (
            (factors.index.get_level_values(0) >= test_start) &
            (factors.index.get_level_values(0) <= test_end)
        )
        X_test = factors[test_mask]
        y_test = labels[test_mask]
        
        # 去除NaN
        train_valid = ~(X_train.isna().any(axis=1) | y_train.isna())
        X_train = X_train[train_valid]
        y_train = y_train[train_valid]
        
        test_valid = ~(X_test.isna().any(axis=1) | y_test.isna())
        X_test = X_test[test_valid]
        y_test = y_test[test_valid]
        
        self.logger.info(f"Dataset prepared - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, y_train, X_test, y_test
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）
            params: 模型参数
            use_cache: 是否使用缓存
        """
        # 检查缓存
        if use_cache:
            cache_file = os.path.join(self.cache_dir, f"{self.model_id}.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        cached_model = pickle.load(f)
                        self.model = cached_model['model']
                        self.feature_importance = cached_model['feature_importance']
                        self.model_params = cached_model['model_params']
                    self.logger.info(f"Loaded model from cache: {cache_file}")
                    return
                except Exception as e:
                    self.logger.warning(f"Failed to load model from cache: {e}")
        
        if params is None:
            params = self._get_default_params()
        
        self.model_params = params
        self.logger.info(f"Training {self.model_type} model with params: {params}")
        
        # 线性模型
        if self.model_type in ["linear", "ridge"]:
            from sklearn.linear_model import Ridge
            self.model = Ridge(**params)
            self.model.fit(X_train, y_train)
            
            self.feature_importance = pd.Series(
                np.abs(self.model.coef_),
                index=X_train.columns
            ).sort_values(ascending=False)
            
        elif self.model_type == "lasso":
            from sklearn.linear_model import Lasso
            self.model = Lasso(**params)
            self.model.fit(X_train, y_train)
            
            self.feature_importance = pd.Series(
                np.abs(self.model.coef_),
                index=X_train.columns
            ).sort_values(ascending=False)
            
        elif self.model_type == "elasticnet":
            from sklearn.linear_model import ElasticNet
            self.model = ElasticNet(**params)
            self.model.fit(X_train, y_train)
            
            self.feature_importance = pd.Series(
                np.abs(self.model.coef_),
                index=X_train.columns
            ).sort_values(ascending=False)
            
        elif self.model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            # 将回归问题转换为分类问题（收益率正负）
            y_train_binary = (y_train > 0).astype(int)
            self.model = LogisticRegression(**params)
            self.model.fit(X_train, y_train_binary)
            
            self.feature_importance = pd.Series(
                np.abs(self.model.coef_[0]),
                index=X_train.columns
            ).sort_values(ascending=False)
        
        # 基于树的模型
        elif self.model_type == "lightgbm":
            import lightgbm as lgb
            
            train_data = lgb.Dataset(X_train, label=y_train)
            
            if X_val is not None and y_val is not None:
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                self.model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[train_data, val_data],
                    num_boost_round=1000,
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
            else:
                self.model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=100
                )
            
            self.feature_importance = pd.Series(
                self.model.feature_importance(),
                index=X_train.columns
            ).sort_values(ascending=False)
        
        elif self.model_type == "xgboost":
            import xgboost as xgb
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            
            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val)
                evals = [(dtrain, 'train'), (dval, 'val')]
                self.model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=1000,
                    evals=evals,
                    early_stopping_rounds=50,
                    verbose_eval=False
                )
            else:
                self.model = xgb.train(params, dtrain, num_boost_round=100)
            
            self.feature_importance = pd.Series(
                self.model.get_score(importance_type='weight'),
                index=X_train.columns
            ).sort_values(ascending=False)
            
        elif self.model_type == "catboost":
            try:
                from catboost import CatBoostRegressor
                self.model = CatBoostRegressor(**params, verbose=False)
                self.model.fit(X_train, y_train)
                
                self.feature_importance = pd.Series(
                    self.model.get_feature_importance(),
                    index=X_train.columns
                ).sort_values(ascending=False)
            except ImportError:
                raise ImportError("CatBoost not installed. Install with: pip install catboost")
            
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(**params)
            self.model.fit(X_train, y_train)
            
            self.feature_importance = pd.Series(
                self.model.feature_importances_,
                index=X_train.columns
            ).sort_values(ascending=False)
            
        elif self.model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(**params)
            self.model.fit(X_train, y_train)
            
            self.feature_importance = pd.Series(
                self.model.feature_importances_,
                index=X_train.columns
            ).sort_values(ascending=False)
            
        elif self.model_type == "decision_tree":
            from sklearn.tree import DecisionTreeRegressor
            self.model = DecisionTreeRegressor(**params)
            self.model.fit(X_train, y_train)
            
            self.feature_importance = pd.Series(
                self.model.feature_importances_,
                index=X_train.columns
            ).sort_values(ascending=False)
        
        # 支持向量机
        elif self.model_type in ["svm", "svr"]:
            from sklearn.svm import SVR
            self.model = SVR(**params)
            self.model.fit(X_train, y_train)
            
            # SVM没有直接的特征重要性，使用基于权重的近似方法
            if hasattr(self.model, 'coef_') and self.model.coef_ is not None:
                self.feature_importance = pd.Series(
                    np.abs(self.model.coef_[0]),
                    index=X_train.columns
                ).sort_values(ascending=False)
            else:
                # 对于非线性SVM，使用基于排列的重要性
                self.feature_importance = self._calculate_permutation_importance(X_train, y_train)
        
        # 朴素贝叶斯
        elif self.model_type == "naive_bayes":
            from sklearn.naive_bayes import GaussianNB
            # 将回归问题转换为分类问题（收益率正负）
            y_train_binary = (y_train > 0).astype(int)
            self.model = GaussianNB(**params)
            self.model.fit(X_train, y_train_binary)
            
            # 朴素贝叶斯没有直接的特征重要性
            self.feature_importance = self._calculate_permutation_importance(X_train, y_train_binary)
        
        # K-最近邻
        elif self.model_type == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            self.model = KNeighborsRegressor(**params)
            self.model.fit(X_train, y_train)
            
            # KNN没有直接的特征重要性
            self.feature_importance = self._calculate_permutation_importance(X_train, y_train)
            
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        self.logger.info(f"Model training completed. Feature importance calculated for top 5 features: "
                        f"{self.feature_importance.head().to_dict()}")
        
        # 保存到缓存
        if use_cache:
            try:
                cache_file = os.path.join(self.cache_dir, f"{self.model_id}.pkl")
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'model': self.model,
                        'feature_importance': self.feature_importance,
                        'model_params': self.model_params
                    }, f)
                self.logger.info(f"Saved model to cache: {cache_file}")
            except Exception as e:
                self.logger.warning(f"Failed to save model to cache: {e}")
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """预测"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.logger.debug(f"Making predictions with {self.model_type} model on data of shape {X.shape}")
        
        # 线性模型
        if self.model_type in ["linear", "ridge", "lasso", "elasticnet"]:
            predictions = self.model.predict(X)
            
        # 分类模型需要特殊处理
        elif self.model_type == "logistic":
            predictions = self.model.predict_proba(X)[:, 1]  # 正类概率
            # 转换为回归预测（概率值）
            predictions = predictions * 2 - 1  # 映射到[-1, 1]范围
            
        elif self.model_type == "naive_bayes":
            predictions = self.model.predict_proba(X)[:, 1]  # 正类概率
            predictions = predictions * 2 - 1  # 映射到[-1, 1]范围
            
        # 基于树的模型
        elif self.model_type == "lightgbm":
            predictions = self.model.predict(X)
        elif self.model_type == "xgboost":
            import xgboost as xgb
            dtest = xgb.DMatrix(X)
            predictions = self.model.predict(dtest)
        elif self.model_type in ["catboost", "random_forest", "gradient_boosting", "decision_tree"]:
            predictions = self.model.predict(X)
            
        # 支持向量机和KNN
        elif self.model_type in ["svm", "svr", "knn"]:
            predictions = self.model.predict(X)
            
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        return pd.Series(predictions, index=X.index)
    
    def _get_default_params(self) -> Dict:
        """获取默认参数"""
        # 线性模型
        if self.model_type in ["linear", "ridge"]:
            return {"alpha": 1.0}
        elif self.model_type == "lasso":
            return {"alpha": 1.0}
        elif self.model_type == "elasticnet":
            return {"alpha": 1.0, "l1_ratio": 0.5}
        elif self.model_type == "logistic":
            return {"C": 1.0, "max_iter": 1000}
            
        # 基于树的模型
        elif self.model_type == "lightgbm":
            return {
                "objective": "regression",
                "metric": "mse",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "verbose": -1
            }
        elif self.model_type == "xgboost":
            return {
                "objective": "reg:squarederror",
                "learning_rate": 0.05,
                "max_depth": 6
            }
        elif self.model_type == "catboost":
            return {
                "iterations": 1000,
                "learning_rate": 0.05,
                "depth": 6,
                "loss_function": "RMSE"
            }
        elif self.model_type == "random_forest":
            return {
                "n_estimators": 100,
                "max_depth": None,
                "random_state": 42
            }
        elif self.model_type == "gradient_boosting":
            return {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3
            }
        elif self.model_type == "decision_tree":
            return {
                "max_depth": None,
                "random_state": 42
            }
            
        # 支持向量机
        elif self.model_type in ["svm", "svr"]:
            return {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale"
            }
            
        # 朴素贝叶斯
        elif self.model_type == "naive_bayes":
            return {}
            
        # K-最近邻
        elif self.model_type == "knn":
            return {
                "n_neighbors": 5,
                "weights": "uniform"
            }
        
        return {}
    
    def _calculate_permutation_importance(self, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5) -> pd.Series:
        """
        计算排列重要性（对于没有内置特征重要性的模型）
        
        Args:
            X: 特征数据
            y: 标签数据
            n_repeats: 重复次数
            
        Returns:
            特征重要性Series
        """
        from sklearn.metrics import mean_squared_error
        import numpy as np
        
        baseline_score = mean_squared_error(y, self.predict(X))
        feature_importance = {}
        
        for feature in X.columns:
            X_permuted = X.copy()
            original_values = X_permuted[feature].copy()
            
            # 多次排列并计算平均影响
            scores = []
            for _ in range(n_repeats):
                X_permuted[feature] = np.random.permutation(original_values)
                permuted_score = mean_squared_error(y, self.predict(X_permuted))
                scores.append(permuted_score - baseline_score)
            
            feature_importance[feature] = np.mean(scores)
        
        return pd.Series(feature_importance).sort_values(ascending=False)