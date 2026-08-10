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
        
        # (Unix Timestamp has been removed as per user request)
        
        # sparse_output=True (default) keeps OHE output as a sparse matrix inside the
        # ColumnTransformer, avoiding a dense 5M×44 allocation during transformation.
        # The combined result is densified once, cast to float32, at the end of fit_transform_edges.
        self.ohe = OneHotEncoder(handle_unknown='ignore')
        
        # Separation of numeric features based on the requested scaling logic
        self.amount_cols = ['Amount Received', 'Amount Paid']
        self.time_cols = []
        
        # LOW cardinality categorical features (One-Hot Encoded)
        self.categorical_cols = ['Receiving Currency', 'Payment Currency', 
                                 'Payment Format', 'Day_of_Week']
        
        # Cyclic features that bypass scaling/encoding
        self.passthrough_cols = ['Hour_Sin', 'Hour_Cos', 'Week']
        
        # The ColumnTransformer handles the dropping of excluded columns.
        # Original 'Timestamp', 'From Bank', and 'To Bank' are permanently removed.
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('amounts', self.amount_scaler, self.amount_cols),
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
        # Removed as per request
        
        # 2. Day of the week (0=Monday, 6=Sunday)
        df_temp['Day_of_Week'] = df_temp['Timestamp'].dt.dayofweek
        
        # 3. Time of day (Cyclic encoding)
        hours = df_temp['Timestamp'].dt.hour
        df_temp['Hour_Sin'] = np.sin(2 * np.pi * hours / 24)
        df_temp['Hour_Cos'] = np.cos(2 * np.pi * hours / 24)
        
        # 4. Week (weeks since the first transaction)
        min_timestamp = df_temp['Timestamp'].min()
        df_temp['Week'] = (df_temp['Timestamp'] - min_timestamp).dt.days // 7
        
        return df_temp

    def fit_transform_edges(self, df: pd.DataFrame, train_mask: pd.Series = None) -> pd.DataFrame:
        """
        Engineers time features, applies scaling/OHE, and drops high-cardinality/old features.
        
        Args:
            df (pd.DataFrame): Raw transactions DataFrame.
            train_mask (pd.Series, optional): Boolean mask indicating training edges.
            
        Returns:
            pd.DataFrame: Transformed edges containing the new features, structural IDs, and labels.
        """
        print("Extracting time features (Unix time, Cyclic Hour, Day of Week, Week)...")
        df_engineered = self._engineer_time_features(df)
        
        print(f"Applying edge transformation (Amounts: {self.scaler_type}, Time: standard)...")
        # Apply transformations (this step drops Timestamp, From Bank, To Bank)
        # Cast to float32 BEFORE toarray() to avoid a massive float64 dense matrix memory spike
        if train_mask is not None:
            self.preprocessor.fit(df_engineered[train_mask])
            processed_array = self.preprocessor.transform(df_engineered)
        else:
            processed_array = self.preprocessor.fit_transform(df_engineered)
        processed_array = processed_array.astype('float32')
        if hasattr(processed_array, 'toarray'):
            processed_array = processed_array.toarray()
        
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
        processed_edges_df['Timestamp'] = df_engineered['Timestamp'].values
        
        return processed_edges_df

    def get_node_features(self, processed_edges_df: pd.DataFrame, train_mask: pd.Series = None) -> pd.DataFrame:
        """
        Aggregates edge features to create node features.
        Implements Equation 1 (node features) and Equation 2 (node labels) from the GAGNN paper.
        
        Args:
            processed_edges_df (pd.DataFrame): Output from fit_transform_edges.
            train_mask (pd.Series, optional): Boolean mask indicating training edges.
            
        Returns:
            pd.DataFrame: DataFrame containing the aggregated features and labels for each node.
        """
        print("Aggregating node features and extracting ground truth labels...")
        
        # Filter out Unix_Timestamp from the features to aggregate
        features_to_aggregate = [f for f in self.feature_names_ if f != 'Unix_Timestamp']
        
        all_nodes = pd.concat([processed_edges_df['Account'], processed_edges_df['Account.1']]).unique()
        
        # 1. Feature aggregation (Training edges only if train_mask provided)
        if train_mask is not None:
            df_to_agg_features = processed_edges_df[train_mask]
        else:
            df_to_agg_features = processed_edges_df
            
        # Group outgoing edges by sender (Node)
        out_grouped_feat = df_to_agg_features.groupby('Account')
        out_sum_feat = out_grouped_feat[features_to_aggregate].sum()
        out_count_feat = out_grouped_feat.size()
        
        # Group incoming edges by receiver (Node)
        in_grouped_feat = df_to_agg_features.groupby('Account.1')
        in_sum_feat = in_grouped_feat[features_to_aggregate].sum()
        in_count_feat = in_grouped_feat.size()
        
        # Combine sums and counts (using fill_value=0 handles nodes that only send or only receive)
        total_sum_feat = out_sum_feat.add(in_sum_feat, fill_value=0)
        total_count_feat = out_count_feat.add(in_count_feat, fill_value=0)
        
        # Calculate the arithmetic mean of the features
        node_features = total_sum_feat.div(total_count_feat, axis=0)
        node_features = node_features.reindex(all_nodes, fill_value=0.0)
        
        # 2. Label aggregation (All edges to retain ground truth for evaluation)
        out_grouped_label = processed_edges_df.groupby('Account')
        out_sum_label = out_grouped_label[['Is Laundering']].sum()
        out_count_label = out_grouped_label.size()
        
        in_grouped_label = processed_edges_df.groupby('Account.1')
        in_sum_label = in_grouped_label[['Is Laundering']].sum()
        in_count_label = in_grouped_label.size()
        
        total_sum_label = out_sum_label.add(in_sum_label, fill_value=0)
        total_count_label = out_count_label.add(in_count_label, fill_value=0)
        
        node_labels = total_sum_label.div(total_count_label, axis=0)
        node_labels = node_labels.reindex(all_nodes, fill_value=0.0)
        
        # Combine features and labels
        node_features['Is Laundering'] = node_labels['Is Laundering']
        
        return node_features