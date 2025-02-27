import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler
import os

class HealthInsuranceRecommender:
    def __init__(self):
        self.user_scaler = StandardScaler()
        self.policy_scaler = StandardScaler()
        self.policies = None
        self.model_path = 'trained_model.pth'
        self.user_scaler_path = 'user_scaler.save'
        self.policy_scaler_path = 'policy_scaler.save'
        
        # Create user model
        self.user_model = self.create_user_model()
        
        # Create policy model
        self.policy_model = self.create_policy_model()

    def create_user_model(self):
        model = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        return model

    def create_policy_model(self):
        model = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        return model

    def train(self, users, policies, epochs=50):
        self.policies = policies
        
        # Preprocess features
        user_features, policy_features = self.preprocess_data(users, policies)
        
        # Scale features
        user_features_scaled = self.user_scaler.fit_transform(user_features)
        policy_features_scaled = self.policy_scaler.fit_transform(policy_features)
        
        # Create all possible user-policy pairs
        user_indices = []
        policy_indices = []
        user_features_list = []
        policy_features_list = []
        
        for i in range(len(users)):
            for j in range(len(policies)):
                user_indices.append(i)
                policy_indices.append(j)
                user_features_list.append(user_features_scaled[i])
                policy_features_list.append(policy_features_scaled[j])
        
        # Convert to tensors
        user_features_tensor = torch.tensor(user_features_list, dtype=torch.float32)
        policy_features_tensor = torch.tensor(policy_features_list, dtype=torch.float32)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(list(self.user_model.parameters()) + list(self.policy_model.parameters()), lr=0.001)
        
        # Training loop
        for epoch in range(epochs):
            self.user_model.train()
            self.policy_model.train()
            
            optimizer.zero_grad()
            user_embeddings = self.user_model(user_features_tensor)
            policy_embeddings = self.policy_model(policy_features_tensor)
            loss = criterion(user_embeddings, policy_embeddings)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')
        
        self.save_model()

    def save_model(self):
        """Save trained model and scalers"""
        try:
            torch.save(self.user_model.state_dict(), 'user_model.pth')
            torch.save(self.policy_model.state_dict(), 'policy_model.pth')
            joblib.dump(self.user_scaler, self.user_scaler_path)
            joblib.dump(self.policy_scaler, self.policy_scaler_path)
            print("Model and scalers saved successfully")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False

    def load_model(self):
        """Load trained model and scalers"""
        try:
            self.user_model.load_state_dict(torch.load('user_model.pth'))
            self.policy_model.load_state_dict(torch.load('policy_model.pth'))
            self.user_scaler = joblib.load(self.user_scaler_path)
            self.policy_scaler = joblib.load(self.policy_scaler_path)
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False

    def get_recommendations(self, user_data, k=3):
        """Get top-k policy recommendations for a user"""
        try:
            # Process user data
            user_features = self.preprocess_data([user_data])
            user_features_scaled = self.user_scaler.transform(user_features)
            
            # Get user embedding
            user_embedding = self.user_model(torch.tensor(user_features_scaled, dtype=torch.float32))
            user_embedding = torch.nn.functional.normalize(user_embedding, dim=1)
            
            # Get policy embeddings
            policy_features = self.preprocess_data([], self.policies)[1]
            policy_features_scaled = self.policy_scaler.transform(policy_features)
            policy_embeddings = self.policy_model(torch.tensor(policy_features_scaled, dtype=torch.float32))
            policy_embeddings = torch.nn.functional.normalize(policy_embeddings, dim=1)
            
            # Calculate base scores using cosine similarity
            base_scores = torch.matmul(user_embedding, policy_embeddings.T)
            
            # Initialize final scores array
            final_scores = torch.zeros(len(self.policies))
            
            # Calculate business rule scores for each policy
            for i, policy in enumerate(self.policies):
                score = 50.0  # Start with base score of 50%
                
                # Budget compatibility (±20%)
                if policy['premium_amount'] <= user_data['budget_amount']:
                    score += 20.0
                else:
                    score -= 20.0
                
                # Coverage match (15%)
                if user_data['maternity_coverage'] and 'maternity' in policy['coverage']:
                    score += 15.0
                
                # Insurer rating (10%)
                if user_data['insurer_rating_preference'] and policy['insurer_rating'] > 0.85:
                    score += 10.0
                
                # Waiting period (5%)
                if policy['waiting_period'] <= 30:
                    score += 5.0
                
                # Combine with model score (normalized to ±20%)
                model_contribution = (base_scores[0][i].item() + 1) * 10  # Convert from [-1,1] to [0,20]
                
                # Final score
                final_scores[i] = min(max(score + model_contribution, 0), 100)
            
            # Get top-k indices
            top_indices = torch.argsort(final_scores, descending=True)[:k]
            
            # Format recommendations
            recommendations = []
            for idx in top_indices:
                policy = self.policies[int(idx)].copy()
                policy['confidence'] = float(final_scores[idx])
                recommendations.append(policy)
            
            return recommendations
            
        except Exception as e:
            print(f"Error in get_recommendations: {str(e)}")
            return []

    def preprocess_data(self, users, policies=None):
        """Preprocess user and policy data"""
        if users:
            user_features = pd.DataFrame({
                'age': [float(u['age']) for u in users],
                'maternity_coverage': [float(u['maternity_coverage']) for u in users],
                'budget': [float(u['budget']) for u in users],
                'insurer_rating_preference': [float(u['insurer_rating_preference']) for u in users]
            })
        
        if policies is None:
            return user_features
        
        policy_features = pd.DataFrame({
            'premium': [float(p['premium']) for p in policies],
            'waiting_period': [float(p['waiting_period'])/180.0 for p in policies],
            'coverage_count': [float(len(p['coverage']))/12.0 for p in policies],
            'insurer_rating': [float(p['insurer_rating']) for p in policies]
        })
        
        if not users:
            return None, policy_features
            
        return user_features, policy_features