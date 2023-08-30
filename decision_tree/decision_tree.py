import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class DecisionTree:
    
    def __init__(self):
        self.decision_tree = []
    
    def fit(self, X, y):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """

        self.decision_tree = []
        
        # If all examples are positive, return a single node tree with label +
        # If all examples are negative, return a single node tree with label -

        # If number of predicting attributes is empty, return a single node 
        # tree with label = most common value of the target attribute.
        
        # Find initial tree node
        root_node = self.get_largest_information_gain(X, y)

        # Insert labels in main dataframe
        X[y.name] = y

        # Start recursive tree building algorithm
        self.split(root_node, X, y.name, [])
    
    def split(self, node, X, target, current_rules):

        # Conditions for leaf nodes:

        # Only one unique value
        if X[target].nunique() == 1:
            self.decision_tree.append((current_rules, X[target].unique()[0]))
        
        # No more attributes to be selected
        elif len(X.columns) == 2:
            self.decision_tree.append((current_rules, X[target].unique()[0]))

        # There are no more rows in the subset
        elif X.empty:
            self.decision_tree.append((current_rules, X[target].mode()[0]))

        # Node is eligible to split
        else:
            # Calculate branches and prepare data
            branch_data_list = [group for _, group in X.groupby(node)]

            # Create all possible branches
            for branch_data in branch_data_list:

                # Update rules for taking branch
                new_rules = current_rules.copy()
                new_rules.append((node, branch_data[node].unique()[0]))

                # Delete current node attribute from dataframe
                branch_data.drop(columns=node, inplace=True)
                branch_data.reset_index(drop=True, inplace=True)

                # Find best attribute for next node
                branch_node = self.get_largest_information_gain(branch_data.drop(columns=target), branch_data[target])

                # Create node at branch
                self.split(branch_node, branch_data, target, new_rules.copy())

        
    def get_largest_information_gain(self, X, y):

        # Calculate inital entropy
        total_entropy = entropy(y.value_counts())

        # Get all attributes
        attributes = X.columns.tolist()

        # Store IG for each attribute
        information_gains = {}

        # Calculate all the weighted entropies
        weighted_entropies = []

        # Calculating information gain for all attributes
        for attribute in attributes:
            # Calculating probability of each value to occur
            unique_values = X[attribute].unique()
            weights = X[attribute].value_counts(normalize=True).reindex(unique_values).fillna(0)

            # Calculating entropies of all the subset rows for each unique value
            entropies = []
            for value in unique_values:
                subset = X[X[attribute] == value]
                label_list = y.iloc[subset.index].tolist()
                counts_list = [label_list.count(label) for label in set(label_list)]
                entropies.append(entropy(np.array(counts_list)))

            # Compute weighted entropy for the attribute
            weighted_entropy = sum(weights[v] * ent for v, ent in zip(unique_values, entropies))

            # Compute information gain for the attribute
            information_gains[attribute] = total_entropy - weighted_entropy
        
        # Return attribute with highest information gain
        return max(information_gains, key=information_gains.get)
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        rules = self.get_rules()
        predictions = []

        # Iterating through each sample in X
        for _, row in X.iterrows():  
            prediction = None
            
            # Checking each rule
            for rule, label in rules:  
                if all(row[feature] == value for feature, value in rule):  
                    # All conditions in a rule are satisfied
                    prediction = label
                    break

            predictions.append(prediction)

        return np.array(predictions)
    
    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        
        return self.decision_tree


# --- Some utility functions 
    
def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))



