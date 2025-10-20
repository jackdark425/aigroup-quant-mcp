"""
深度学习模型 - LSTM/GRU/Transformer
参考Qlib实现的深度学习模型
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Union, Tuple
from pathlib import Path
import copy


class LSTMModel:
    """
    LSTM模型用于股票价格预测
    
    Parameters:
    -----------
    d_feat : int
        输入特征维度
    hidden_size : int
        LSTM隐藏层大小
    num_layers : int
        LSTM层数
    dropout : float
        Dropout比率
    n_epochs : int
        训练轮数
    lr : float
        学习率
    batch_size : int
        批次大小
    early_stop : int
        早停轮数
    optimizer : str
        优化器类型 ('adam', 'sgd')
    loss_fn : str
        损失函数 ('mse', 'mae')
    device : str
        设备 ('cuda', 'cpu')
    """
    
    def __init__(
        self,
        d_feat: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        n_epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 800,
        early_stop: int = 20,
        optimizer: str = 'adam',
        loss_fn: str = 'mse',
        device: str = 'cpu',
        seed: Optional[int] = None
    ):
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer_name = optimizer.lower()
        self.loss_fn_name = loss_fn.lower()
        self.device = torch.device(device)
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        # 创建模型
        self.model = _LSTMNet(
            d_feat=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        # 创建优化器
        if self.optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif self.optimizer_name == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"不支持的优化器: {optimizer}")
            
        self.fitted = False
        
    def _get_loss_fn(self):
        """获取损失函数"""
        if self.loss_fn_name == 'mse':
            return nn.MSELoss()
        elif self.loss_fn_name == 'mae':
            return nn.L1Loss()
        else:
            raise ValueError(f"不支持的损失函数: {self.loss_fn_name}")
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None
    ) -> Dict:
        """
        训练模型
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            训练特征
        y_train : pd.Series
            训练标签
        X_valid : pd.DataFrame, optional
            验证特征
        y_valid : pd.Series, optional
            验证标签
            
        Returns:
        --------
        Dict
            训练历史
        """
        X_train_values = X_train.values.astype(np.float32)
        y_train_values = y_train.values.astype(np.float32)
        
        if X_valid is not None and y_valid is not None:
            X_valid_values = X_valid.values.astype(np.float32)
            y_valid_values = y_valid.values.astype(np.float32)
        else:
            X_valid_values = None
            y_valid_values = None
            
        loss_fn = self._get_loss_fn()
        
        best_loss = np.inf
        best_epoch = 0
        best_state = None
        stop_steps = 0
        
        train_losses = []
        valid_losses = []
        
        for epoch in range(self.n_epochs):
            # 训练
            self.model.train()
            train_loss = self._train_epoch(X_train_values, y_train_values, loss_fn)
            train_losses.append(train_loss)
            
            # 验证
            if X_valid_values is not None:
                self.model.eval()
                valid_loss = self._eval_epoch(X_valid_values, y_valid_values, loss_fn)
                valid_losses.append(valid_loss)
                
                # 早停
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.model.state_dict())
                    stop_steps = 0
                else:
                    stop_steps += 1
                    if stop_steps >= self.early_stop:
                        print(f"早停于第 {epoch} 轮，最佳轮次: {best_epoch}")
                        break
                        
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.n_epochs} - "
                          f"Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.n_epochs} - Train Loss: {train_loss:.6f}")
        
        # 加载最佳模型
        if best_state is not None:
            self.model.load_state_dict(best_state)
            
        self.fitted = True
        
        return {
            'train_loss': train_losses,
            'valid_loss': valid_losses,
            'best_epoch': best_epoch,
            'best_loss': best_loss
        }
    
    def _train_epoch(self, X, y, loss_fn) -> float:
        """训练一个epoch"""
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        
        total_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            X_batch = torch.FloatTensor(X[batch_indices]).to(self.device)
            y_batch = torch.FloatTensor(y[batch_indices]).to(self.device)
            
            self.optimizer.zero_grad()
            
            pred = self.model(X_batch)
            loss = loss_fn(pred, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        return total_loss / n_batches
    
    def _eval_epoch(self, X, y, loss_fn) -> float:
        """评估一个epoch"""
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                X_batch = torch.FloatTensor(X[i:i + self.batch_size]).to(self.device)
                y_batch = torch.FloatTensor(y[i:i + self.batch_size]).to(self.device)
                
                pred = self.model(X_batch)
                loss = loss_fn(pred, y_batch)
                
                total_loss += loss.item()
                n_batches += 1
                
        return total_loss / n_batches
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        预测
        
        Parameters:
        -----------
        X : pd.DataFrame
            输入特征
            
        Returns:
        --------
        pd.Series
            预测结果
        """
        if not self.fitted:
            raise RuntimeError("模型尚未训练")
            
        self.model.eval()
        X_values = X.values.astype(np.float32)
        
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X_values), self.batch_size):
                X_batch = torch.FloatTensor(X_values[i:i + self.batch_size]).to(self.device)
                pred = self.model(X_batch)
                predictions.append(pred.cpu().numpy())
                
        predictions = np.concatenate(predictions)
        return pd.Series(predictions, index=X.index)
    
    def save(self, path: Union[str, Path]):
        """保存模型"""
        torch.save({
            'model_state': self.model.state_dict(),
            'config': {
                'd_feat': self.d_feat,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
            }
        }, path)
        
    def load(self, path: Union[str, Path]):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.fitted = True


class GRUModel(LSTMModel):
    """
    GRU模型用于股票价格预测
    
    参数与LSTMModel相同
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 替换为GRU网络
        self.model = _GRUNet(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # 重新创建优化器
        if self.optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_name == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)


class _LSTMNet(nn.Module):
    """LSTM网络架构"""
    
    def __init__(
        self,
        d_feat: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        
        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc_out = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        前向传播
        
        Parameters:
        -----------
        x : torch.Tensor
            形状 [batch_size, seq_len * d_feat] 或 [batch_size, seq_len, d_feat]
            
        Returns:
        --------
        torch.Tensor
            预测值，形状 [batch_size]
        """
        # 如果输入是2D的，需要reshape
        if x.dim() == 2:
            batch_size = x.size(0)
            # 检查是否可以reshape为序列格式
            if x.size(1) % self.d_feat == 0:
                seq_len = x.size(1) // self.d_feat
                x = x.reshape(batch_size, seq_len, self.d_feat)
            else:
                # 如果无法reshape为序列，则视为单时间步特征
                # 添加序列维度: [batch_size, features] -> [batch_size, 1, features]
                x = x.unsqueeze(1)
        
        # LSTM: [batch_size, seq_len, d_feat] -> [batch_size, seq_len, hidden_size]
        out, _ = self.rnn(x)
        
        # 取最后一个时间步
        out = out[:, -1, :]  # [batch_size, hidden_size]
        
        # 全连接层
        out = self.fc_out(out)  # [batch_size, 1]
        
        return out.squeeze(-1)  # [batch_size]


class _GRUNet(nn.Module):
    """GRU网络架构"""
    
    def __init__(
        self,
        d_feat: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        
        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc_out = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        前向传播
        
        Parameters:
        -----------
        x : torch.Tensor
            形状 [batch_size, seq_len * d_feat] 或 [batch_size, seq_len, d_feat]
            
        Returns:
        --------
        torch.Tensor
            预测值，形状 [batch_size]
        """
        # 如果输入是2D的，需要reshape
        if x.dim() == 2:
            batch_size = x.size(0)
            # 检查是否可以reshape为序列格式
            if x.size(1) % self.d_feat == 0:
                seq_len = x.size(1) // self.d_feat
                x = x.reshape(batch_size, seq_len, self.d_feat)
            else:
                # 如果无法reshape为序列，则视为单时间步特征
                # 添加序列维度: [batch_size, features] -> [batch_size, 1, features]
                x = x.unsqueeze(1)
        
        # GRU: [batch_size, seq_len, d_feat] -> [batch_size, seq_len, hidden_size]
        out, _ = self.rnn(x)
        
        # 取最后一个时间步
        out = out[:, -1, :]  # [batch_size, hidden_size]
        
        # 全连接层
        out = self.fc_out(out)  # [batch_size, 1]
        
        return out.squeeze(-1)  # [batch_size]


class TransformerModel:
    """
    Transformer模型用于股票价格预测
    
    Parameters:
    -----------
    d_feat : int
        输入特征维度
    d_model : int
        Transformer模型维度
    nhead : int
        多头注意力头数
    num_layers : int
        Transformer层数
    dropout : float
        Dropout比率
    n_epochs : int
        训练轮数
    lr : float
        学习率
    batch_size : int
        批次大小
    early_stop : int
        早停轮数
    optimizer : str
        优化器类型
    loss_fn : str
        损失函数
    device : str
        设备
    """
    
    def __init__(
        self,
        d_feat: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 800,
        early_stop: int = 20,
        optimizer: str = 'adam',
        loss_fn: str = 'mse',
        device: str = 'cpu',
        seed: Optional[int] = None
    ):
        self.d_feat = d_feat
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer_name = optimizer.lower()
        self.loss_fn_name = loss_fn.lower()
        self.device = torch.device(device)
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        # 创建模型
        self.model = _TransformerNet(
            d_feat=d_feat,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        # 创建优化器
        if self.optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif self.optimizer_name == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"不支持的优化器: {optimizer}")
            
        self.fitted = False
        
    def _get_loss_fn(self):
        """获取损失函数"""
        if self.loss_fn_name == 'mse':
            return nn.MSELoss()
        elif self.loss_fn_name == 'mae':
            return nn.L1Loss()
        else:
            raise ValueError(f"不支持的损失函数: {self.loss_fn_name}")
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None
    ) -> Dict:
        """训练模型（与LSTMModel相同的接口）"""
        X_train_values = X_train.values.astype(np.float32)
        y_train_values = y_train.values.astype(np.float32)
        
        if X_valid is not None and y_valid is not None:
            X_valid_values = X_valid.values.astype(np.float32)
            y_valid_values = y_valid.values.astype(np.float32)
        else:
            X_valid_values = None
            y_valid_values = None
            
        loss_fn = self._get_loss_fn()
        
        best_loss = np.inf
        best_epoch = 0
        best_state = None
        stop_steps = 0
        
        train_losses = []
        valid_losses = []
        
        for epoch in range(self.n_epochs):
            # 训练
            self.model.train()
            train_loss = self._train_epoch(X_train_values, y_train_values, loss_fn)
            train_losses.append(train_loss)
            
            # 验证
            if X_valid_values is not None:
                self.model.eval()
                valid_loss = self._eval_epoch(X_valid_values, y_valid_values, loss_fn)
                valid_losses.append(valid_loss)
                
                # 早停
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.model.state_dict())
                    stop_steps = 0
                else:
                    stop_steps += 1
                    if stop_steps >= self.early_stop:
                        print(f"早停于第 {epoch} 轮，最佳轮次: {best_epoch}")
                        break
                        
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.n_epochs} - "
                          f"Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.n_epochs} - Train Loss: {train_loss:.6f}")
        
        # 加载最佳模型
        if best_state is not None:
            self.model.load_state_dict(best_state)
            
        self.fitted = True
        
        return {
            'train_loss': train_losses,
            'valid_loss': valid_losses,
            'best_epoch': best_epoch,
            'best_loss': best_loss
        }
    
    def _train_epoch(self, X, y, loss_fn) -> float:
        """训练一个epoch"""
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        
        total_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            X_batch = torch.FloatTensor(X[batch_indices]).to(self.device)
            y_batch = torch.FloatTensor(y[batch_indices]).to(self.device)
            
            self.optimizer.zero_grad()
            
            pred = self.model(X_batch)
            loss = loss_fn(pred, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        return total_loss / n_batches
    
    def _eval_epoch(self, X, y, loss_fn) -> float:
        """评估一个epoch"""
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                X_batch = torch.FloatTensor(X[i:i + self.batch_size]).to(self.device)
                y_batch = torch.FloatTensor(y[i:i + self.batch_size]).to(self.device)
                
                pred = self.model(X_batch)
                loss = loss_fn(pred, y_batch)
                
                total_loss += loss.item()
                n_batches += 1
                
        return total_loss / n_batches
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """预测"""
        if not self.fitted:
            raise RuntimeError("模型尚未训练")
            
        self.model.eval()
        X_values = X.values.astype(np.float32)
        
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X_values), self.batch_size):
                X_batch = torch.FloatTensor(X_values[i:i + self.batch_size]).to(self.device)
                pred = self.model(X_batch)
                predictions.append(pred.cpu().numpy())
                
        predictions = np.concatenate(predictions)
        return pd.Series(predictions, index=X.index)
    
    def save(self, path: Union[str, Path]):
        """保存模型"""
        torch.save({
            'model_state': self.model.state_dict(),
            'config': {
                'd_feat': self.d_feat,
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
            }
        }, path)
        
    def load(self, path: Union[str, Path]):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.fitted = True


class _TransformerNet(nn.Module):
    """Transformer网络架构"""
    
    def __init__(
        self,
        d_feat: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_feat = d_feat
        self.d_model = d_model
        
        # 输入投影
        self.input_proj = nn.Linear(d_feat, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 输出层
        self.fc_out = nn.Linear(d_model, 1)
        
    def forward(self, x):
        """
        前向传播
        
        Parameters:
        -----------
        x : torch.Tensor
            形状 [batch_size, seq_len * d_feat] 或 [batch_size, seq_len, d_feat]
            
        Returns:
        --------
        torch.Tensor
            预测值，形状 [batch_size]
        """
        # 如果输入是2D的，需要reshape
        if x.dim() == 2:
            batch_size = x.size(0)
            # 检查是否可以reshape为序列格式
            if x.size(1) % self.d_feat == 0:
                seq_len = x.size(1) // self.d_feat
                x = x.reshape(batch_size, seq_len, self.d_feat)
            else:
                # 如果无法reshape为序列，则视为单时间步特征
                # 添加序列维度: [batch_size, features] -> [batch_size, 1, features]
                x = x.unsqueeze(1)
        
        # 输入投影: [batch_size, seq_len, d_feat] -> [batch_size, seq_len, d_model]
        x = self.input_proj(x)
        
        # Transformer编码: [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        x = self.transformer(x)
        
        # 取最后一个时间步
        x = x[:, -1, :]  # [batch_size, d_model]
        
        # 输出层
        x = self.fc_out(x)  # [batch_size, 1]
        
        return x.squeeze(-1)  # [batch_size]