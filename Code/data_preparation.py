import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class DataPreparation:
    """
    Class for preparing the IBM AMLSim data for the GAGNN model.
    Handles time feature engineering, edge feature normalization, 
    and their aggregation into node features.
    """
    def __init__(self, scaler_type='robust'):
        """
        Initializes the data preparation class.
        
        Args:
            scaler_type (str): 'robust' to use RobustScaler, 
                               'standard' to use StandardScaler for amounts.
        """
        if scaler_type not in ['robust', 'standard']:
            raise ValueError("scaler_type must be 'robust' or 'standard'")
        
        self.scaler_type = scaler_type
        
        # Scaler specifically for financial amounts (dynamic based on input)
        self.amount_scaler = RobustScaler() if scaler_type == 'robust' else StandardScaler()
        
        # Scaler specifically for Unix Timestamp (always standard)
        self.time_scaler = StandardScaler()
        
        # sparse_output=False ensures a dense array ready for the neural network
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Separation of numeric features based on the requested scaling logic
        self.amount_cols = ['Amount Received', 'Amount Paid']
        self.time_cols = ['Unix_Timestamp']
        
        # LOW cardinality categorical features (One-Hot Encoded)
        self.categorical_cols = ['Receiving Currency', 'Payment Currency', 
                                 'Payment Format', 'Day_of_Week']
        
        # Cyclic features that bypass scaling/encoding
        self.passthrough_cols = ['Hour_Sin', 'Hour_Cos']
        
        # The ColumnTransformer handles the dropping of excluded columns.
        # Original 'Timestamp', 'From Bank', and 'To Bank' are permanently removed.
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('amounts', self.amount_scaler, self.amount_cols),
                ('time', self.time_scaler, self.time_cols),
                ('cat', self.ohe, self.categorical_cols),
                ('pass', 'passthrough', self.passthrough_cols)
            ],
            remainder='drop'
        )
        
        self.feature_names_ = None

    def _engineer_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts engineered features from the Timestamp field.
        
        Args:
            df (pd.DataFrame): Raw DataFrame.
            
        Returns:
            pd.DataFrame: DataFrame with the new time features added.
        """
        df_temp = df.copy()
        
        # Ensure Timestamp is a datetime object
        if not pd.api.types.is_datetime64_any_dtype(df_temp['Timestamp']):
            df_temp['Timestamp'] = pd.to_datetime(df_temp['Timestamp'])
            
        # 1. Unix Timestamp (seconds since 1970)
        df_temp['Unix_Timestamp'] = df_temp['Timestamp'].astype('int64') // 10**9
        
        # 2. Day of the week (0=Monday, 6=Sunday)
        df_temp['Day_of_Week'] = df_temp['Timestamp'].dt.dayofweek
        
        # 3. Time of day (Cyclic encoding)
        hours = df_temp['Timestamp'].dt.hour
        df_temp['Hour_Sin'] = np.sin(2 * np.pi * hours / 24)
        df_temp['Hour_Cos'] = np.cos(2 * np.pi * hours / 24)
        
        return df_temp

    def fit_transform_edges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers time features, applies scaling/OHE, and drops high-cardinality/old features.
        
        Args:
            df (pd.DataFrame): Raw transactions DataFrame.
            
        Returns:
            pd.DataFrame: Transformed edges containing the new features, structural IDs, and labels.
        """
        print("Extracting time features (Unix time, Cyclic Hour, Day of Week)...")
        df_engineered = self._engineer_time_features(df)
        
        print(f"Applying edge transformation (Amounts: {self.scaler_type}, Time: standard)...")
        # Apply transformations (this step drops Timestamp, From Bank, To Bank)
        processed_array = self.preprocessor.fit_transform(df_engineered)
        
        # Retrieve feature names to reconstruct the DataFrame in the exact order output by ColumnTransformer
        amount_names = self.amount_cols
        time_names = self.time_cols
        cat_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(self.categorical_cols)
        pass_names = self.passthrough_cols
        
        self.feature_names_ = list(amount_names) + list(time_names) + list(cat_names) + list(pass_names)
        
        # Reconstruct the processed DataFrame for the edges
        processed_edges_df = pd.DataFrame(
            processed_array, 
            columns=self.feature_names_, 
            index=df.index
        )
        
        # Re-integrate structural columns AND the ground truth label
        processed_edges_df['Account'] = df_engineered['Account'].values
        processed_edges_df['Account.1'] = df_engineered['Account.1'].values
        processed_edges_df['Is Laundering'] = df_engineered['Is Laundering'].values
        
        return processed_edges_df

    def get_node_features(self, processed_edges_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates edge features to create node features.
        Implements Equation 1 (node features) and Equation 2 (node labels) from the GAGNN paper.
        
        Args:
            processed_edges_df (pd.DataFrame): Output from fit_transform_edges.
            
        Returns:
            pd.DataFrame: DataFrame containing the aggregated features and labels for each node.
        """
        print("Aggregating node features and extracting ground truth labels...")
        
        # Outgoing edges (The node is the sender: 'Account')
        df_out = processed_edges_df.drop(
            columns=['Account.1']
        ).rename(columns={'Account': 'Node'})
        
        # Incoming edges (The node is the receiver: 'Account.1')
        df_in = processed_edges_df.drop(
            columns=['Account']
        ).rename(columns={'Account.1': 'Node'})
        
        # Merge all connection instances for each node (both incoming and outgoing)
        df_all_incident = pd.concat([df_out, df_in], ignore_index=True)
        
        # Filter out Unix_Timestamp from the features to aggregate
        features_to_aggregate = [f for f in self.feature_names_ if f != 'Unix_Timestamp']
        
        # Group by node
        grouped = df_all_incident.groupby('Node')
        
        # Calculate the arithmetic mean of the features (Equation 1)
        node_features = grouped[features_to_aggregate].mean()
        
        # Calculate node soft labels (Equation 2 variation)
        # The mean of incident edges' labels gives the exact proportion of ML transactions (probability)
        node_features['Is Laundering'] = grouped['Is Laundering'].mean()
        
        return node_features