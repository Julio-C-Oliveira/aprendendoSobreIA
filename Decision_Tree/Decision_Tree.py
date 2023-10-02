import numpy as np # Serve para facilitar as modificações em listas
from sklearn import datasets # Pra obter datasets
from sklearn.model_selection import train_test_split # Só pra dividir o dataset
from sklearn.metrics import mean_squared_error # Calcular o erro médio
from collections import Counter # Contagem de repetição de elementos

"""
Um algoritimo de Decisio Tree e outro de Random Forest, ainda inacabado.
Ambas possuem uma váriavel chamada mode, que serve para escolher entre:
- 1 para Regressão.
- 0 para Classificação.
"""

# Nó

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None



# Árvore

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, num_features=None, mode=0):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.num_features = num_features
        self.root_node = None
        self.iterate = 0
        self.split_criterion = self.general_entropy if mode == 0 else self.variance_reduction
        self.calculate_leaf_value = self._most_common_label if mode == 0 else self._mean_values
        self.accuracy_function = self.classification_accuracy if mode == 0 else self.regression_accuracy
        self.tree_shape = []


    def fit(self, data, label):
        self.num_features = data.shape[1] if not self.num_features else min(data.shape[1],self.num_features)
        self.root_node = self._grow_tree(data, label)


    def _grow_tree(self, data, label, depth=0):
        self.iterate += 1

        num_samples, num_features = data.shape
        num_labels = len(np.unique(label))

        if (depth >= self.max_depth or num_labels <= 1 or num_samples < self.min_samples_split):
            leaf_value = self._calculate_leaf_value(label)
            return Node(value=leaf_value)

        feature_indexs = np.random.choice(num_features, self.num_features, replace=False)

        best_feature, best_threshold = self._best_split(data, label, feature_indexs)

        left_indexs, right_indexs = self._split(data[:, best_feature], best_threshold)

        left = self._grow_tree(data[left_indexs, :], label[left_indexs], depth+1)
        right = self._grow_tree(data[right_indexs, :], label[right_indexs], depth+1)
        return Node(best_feature, best_threshold, left, right)


    def _best_split(self, data, label, feature_indexs):
        best_gain = -1
        split_indexs, split_threshold = None, None

        for feature_index in feature_indexs:
            data_column = data[:, feature_index]
            thresholds = np.unique(data_column)

            for threshold in thresholds:
                gain = self.information_gain(data_column, label, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_index = feature_index
                    split_threshold = threshold

        return split_index, split_threshold

    def information_gain(self, data_column, label, threshold):
        left_indexs, right_indexs = self._split(data_column, threshold)

        if len(left_indexs) == 0 or len(right_indexs) == 0:
            return 0

        left_weight = len(left_indexs) / len(label)
        right_weight = len(right_indexs) / len(label)

        return self.split_criterion(label, left_indexs, right_indexs, left_weight, right_weight)


    def _split(self, data_column, split_threshold):
        left_indexs = np.argwhere(data_column <= split_threshold).flatten()
        right_indexs = np.argwhere(data_column > split_threshold).flatten()
        return left_indexs, right_indexs


    def general_entropy(self, label, left_indexs, right_indexs, left_weight, right_weight):
        label_weight = len(label)
        label_entropy = self.entropy(label)
        left_entropy = self.entropy(label[left_indexs])
        right_entropy = self.entropy(label[right_indexs])
        child_entropy = (left_weight/label_weight) * left_entropy + (right_weight/label_weight) * right_entropy
        return label_entropy - child_entropy

    def entropy(self, label):
        label_weight = len(label)
        histogram = np.bincount(label)
        probabilities = histogram / label_weight
        return -np.sum([probabilitie * np.log(probabilitie) for probabilitie in probabilities if probabilitie>0])

    def variance_reduction(self, label, left_indexs, right_indexs, left_weight, right_weight):
        return np.var(label) - (left_weight * np.var(left_indexs) + right_weight * np.var(right_indexs))


    def _calculate_leaf_value(self, label):
      return self.calculate_leaf_value(label)

    def _most_common_label(self, label):
        counter = Counter(label)
        value = counter.most_common(1)[0][0]
        return value

    def _mean_values(self, label):
        if len(label) > 0: return np.mean(label)
        else: return 0


    def predict(self, data):
        return np.array([self._traverse_tree(x, self.root_node) for x in data])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


    def accuracy(self, test_label, prediction_label):
        return self.accuracy_function(test_label, prediction_label)

    def classification_accuracy(self, test_label, prediction_label):
        accuracy = np.sum(test_label == prediction_label) / len(test_label)
        print(f"Precisão: {accuracy*100:.2f}%")

        return accuracy

    def regression_accuracy(self, test_label, prediction_label):
        from sklearn.metrics import mean_squared_error
        mean_squared_error = np.sqrt(mean_squared_error(test_label, prediction_label))
        print(f"Erro médio: {mean_squared_error:.2f}")

        return mean_squared_error


    def _display_tree(self, node, direction=0):
        self.tree_shape.append([node.threshold, direction])
        if not node.is_leaf_node(): # Retorna os valores necessários e continua
            self._display_tree(node=node.left, direction=1)
            self._display_tree(node=node.right, direction=2)

#    def reconstruct_tree(self, node, index=0): Pretendo adicionar uma função para replicar uma árvore
#        threshold_node = 0
#        left_node = 0
#        right_node = 0

#        self.root_node = Node(threshold=threshold_node, left=left_node, right=right_node)


# Random Forests

class RandomForest:
    def __init__(self, num_trees=10, max_depth=10, min_samples_split=2, num_feature=None, mode=0):
        self.num_trees = num_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.num_features=num_feature
        self.trees = [] # Vai armazenar as árvores
        self.mode = mode
        self.calculate_predict = self._most_common_label if mode == 0 else self._mean_values

    def fit(self, data, label):
        self.trees = []
        for _ in range(self.num_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                num_features=self.num_features, mode=self.mode)

            data_sample, label_sample = self._bootstrap_samples(data, label)
            tree.fit(data_sample, label_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, data, label):
        num_samples = data.shape[0]
        indexs = np.random.choice(num_samples, num_samples, replace=True)
        return data[indexs], label[indexs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _mean_values(self, label):
        if len(label) > 0: return np.mean(label)
        else: return 0

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self.calculate_predict(pred) for pred in tree_preds])
        return predictions


#################################

# Random Forest

data = datasets.load_wine()
#data = datasets.load_breast_cancer()
#data = datasets.load_digits()
#data = datasets.load_iris()

data, label = data.data, data.target

data_train, data_test, label_train, label_test = train_test_split(
    data, label, test_size=0.2, random_state=1234
)

model = RandomForest(num_trees=50, mode=0)
model.fit(data_train, label_train)
predictions = model.predict(data_test)

print("Random Forest:")
forest_result = model.trees[0].accuracy(label_test, predictions)

# Decision Tree

model = DecisionTree(max_depth=15, mode=0)
model.fit(data_train, label_train)

predictions = model.predict(data_test)

print("Decision Tree:")
tree_result = model.accuracy(label_test, predictions)