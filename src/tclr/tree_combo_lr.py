import codecs
from io import StringIO

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from treelib import Tree
from IPython import embed as II


class TreeComboLR:
    node_count = 0
    init_N = 0
    min_mse = float("inf")
    max_mse = -float("inf")

    def __init__(
        self,
        X,
        y,
        min_samples_split=None,
        max_depth=None,
        tree_vars=None,
        reg_vars=None,
        curr_depth=0,
        method="Nelder-Mead",
        feature_names=None,
        response_name=None,
        node_type=None,
        ID=0,
        parent=None,
    ):
        self.N = X.shape[0]
        if isinstance(X, pd.DataFrame):
            self.X = X.values
            self.feats = list(X.columns)
        else:
            self.X = X
            self.feats = list(range(X.shape[1]))

        if isinstance(X, (pd.DataFrame, pd.Series)):
            self.y = y.values
            self.response = y.name
        else:
            self.y = y
            self.response = "y"

        if tree_vars is not None:
            self.tree_vars = [self.feats.index(i) for i in tree_vars]
        else:
            self.tree_vars = list(range(X.shape[1]))

        if reg_vars is not None:
            self.reg_vars = [self.feats.index(i) for i in reg_vars]
        else:
            self.reg_vars = list(range(X.shape[1]))

        self.feats = feature_names if feature_names else self.feats
        self.response = response_name if response_name else self.response

        self.min_samples_split = (
            min_samples_split if min_samples_split else int(self.N * 0.05)
        )
        self.max_depth = max_depth if max_depth else 4
        self.curr_depth = curr_depth
        self.method = method

        self.rule = None

        self.node_type = node_type if node_type else "root"
        self.ID = ID
        self.parent = parent
        if ID == 0:
            TreeComboLR.node_count = 0
            TreeComboLR.min_mse = float("inf")
            TreeComboLR.max_mse = -float("inf")
            TreeComboLR.init_N = self.N

        self.left = None
        self.right = None

        self.best_feat = None
        self.best_val = None

        try:
            self.params = self._solve_regression()
        except np.linalg.LinAlgError as e:
            II()
        yhat = self._predict_regression(self.params)

        self.mse = mean_squared_error(self.y, yhat)
        if self.mse < TreeComboLR.min_mse:
            TreeComboLR.min_mse = self.mse
        if self.mse > TreeComboLR.max_mse:
            TreeComboLR.max_mse = self.mse

    def _solve_regression(self, X=None, y=None, reg_vars=None):
        X = self.X if X is None else X
        y = self.y if y is None else y
        reg_vars = self.reg_vars if reg_vars is None else reg_vars
        X = X[:, reg_vars]
        return np.linalg.inv(X.T @ X) @ (X.T @ y)

    def _predict_regression(self, p, X=None):
        X = self.X if X is None else X
        X = X[:, self.reg_vars]
        return X @ p

    def _split_node_data(self, thresh, feat_id, X=None, y=None):
        X = self.X if X is None else X
        y = self.y if y is None else y

        right = np.argwhere(X[:, feat_id] > thresh)
        left = np.argwhere(X[:, feat_id] <= thresh)

        X_right = X[right[:, 0]]
        X_left = X[left[:, 0]]
        y_right = y[right[:, 0]]
        y_left = y[left[:, 0]]

        return X_left, X_right, y_left, y_right

    def _check_all_entries_same(self, matrix):
        bad_columns = []
        for col in range(matrix.shape[1]):
            result = np.all(matrix[:, col] == matrix[0, col])
            if result:
                bad_columns.append(col)
        return bad_columns

    def _check_all_entries_zero(self, matrix):
        return list(np.where(~matrix.any(axis=0))[0])

    def _get_node_score(self, thresh, feat_id, X=None, y=None):
        X = self.X if X is None else X
        y = self.y if y is None else y

        X_left, X_right, y_left, y_right = self._split_node_data(thresh, feat_id)

        N_left = y_left.shape[0]
        N_right = y_right.shape[0]

        # i do not know if this is optimal or not but it prevents
        # splits with zero in either the left or right side
        if N_left == 0 or N_right == 0:
            return float("inf")
        try:
            p_left = self._solve_regression(X_left, y_left)
            left_reg_vars = self.reg_vars
        except np.linalg.LinAlgError as e:
            bad_columns = self._check_all_entries_zero(X_left)
            bad_features = [self.feats[i] for i in bad_columns]
            print(f"Node {self.ID} Left Split: {e}")
            print(f"Dropping {', '.join(bad_features)} because they are all zero")
            print("Columns of zero create singular matrices due to linear dependence")

            left_reg_vars = list(filter(lambda x: x not in bad_columns, self.reg_vars))
            p_left = self._solve_regression(X_left, y_left, left_reg_vars)
            for col in bad_columns:
                p_left = np.insert(p_left, col, 0.0)

        try:
            p_right = self._solve_regression(X_right, y_right)
            right_reg_vars = self.reg_vars
        except np.linalg.LinAlgError as e:
            bad_columns = self._check_all_entries_zero(X_right)
            bad_features = [self.feats[i] for i in bad_columns]
            print(f"Node {self.ID} Right Split: {e}")
            print(f"Dropping {', '.join(bad_features)} because they are all zero")
            print("Columns of zero create singular matrices due to linear dependence")

            right_reg_vars = list(filter(lambda x: x not in bad_columns, self.reg_vars))
            p_right = self._solve_regression(X_right, y_right, right_reg_vars)
            for col in bad_columns:
                p_right = np.insert(p_right, col, 0.0)

        yhat_left = self._predict_regression(p_left, X_left)
        yhat_right = self._predict_regression(p_right, X_right)

        mse_left = mean_squared_error(y_left, yhat_left)
        mse_right = mean_squared_error(y_right, yhat_right)

        left_score = N_left / y.shape[0] * mse_left
        right_score = N_right / y.shape[0] * mse_right

        return left_score + right_score

    def _optimize_node(self, X=None, y=None):
        X = self.X if X is None else X
        y = self.y if y is None else y

        best_feat = None
        best_val = None
        mse = self.mse

        for feat_id in self.tree_vars:
            opt = minimize(
                self._get_node_score,
                [np.mean(X[:, feat_id])],
                args=(feat_id, X, y),
                method=self.method,
            )

            if opt.fun < mse:
                X_left, X_right, y_left, y_right = self._split_node_data(
                    opt.x[0], feat_id
                )
                if X_left.shape[0] > 0 and X_right.shape[0] > 0:
                    best_feat = feat_id
                    best_val = opt.x[0]
                    mse = opt.fun
                    if mse < TreeComboLR.min_mse:
                        TreeComboLR.min_mse = mse
                    if mse > TreeComboLR.max_mse:
                        TreeComboLR.max_mse = mse

        return best_feat, best_val

    def grow_tree(self):
        N = self.X.shape[0]

        if (self.curr_depth < self.max_depth) and (N >= self.min_samples_split):
            best_feat, best_val = self._optimize_node()
            if best_feat is not None:
                X_left, X_right, y_left, y_right = self._split_node_data(
                    best_val, best_feat
                )

                self.best_feat = best_feat
                self.best_val = best_val
                self.rule = f"{self.feats[best_feat]} > {best_val:.3f}"

                left = TreeComboLR(
                    X=X_left,
                    y=y_left,
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth,
                    curr_depth=self.curr_depth + 1,
                    method=self.method,
                    feature_names=self.feats,
                    response_name=self.response,
                    tree_vars=self.tree_vars,
                    reg_vars=self.reg_vars,
                    node_type="left_node",
                    ID=TreeComboLR.node_count + 1,
                    parent=self.ID,
                )

                TreeComboLR.node_count += 1
                self.left = left
                self.left.grow_tree()

                right = TreeComboLR(
                    X=X_right,
                    y=y_right,
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth,
                    curr_depth=self.curr_depth + 1,
                    method=self.method,
                    feature_names=self.feats,
                    response_name=self.response,
                    tree_vars=self.tree_vars,
                    reg_vars=self.reg_vars,
                    node_type="right_node",
                    ID=TreeComboLR.node_count + 1,
                    parent=self.ID,
                )

                TreeComboLR.node_count += 1
                self.right = right
                self.right.grow_tree()

    def _print_info(self, width=4):
        const = int(self.curr_depth * width ** 1.5)
        spaces = "-" * const

        if self.node_type == "root":
            print("Root")
        else:
            print(f"|{spaces} Split rule: {self.rule}")
        print(f"{' ' * const}   | MSE of the node: {self.mse:.2f}")
        print(f"{' ' * const}   | N obs in the node: {self.X.shape[0]:.0f}")

    def _print_params(self, width=4):
        const = int(self.curr_depth * width ** 1.5)
        param_format = [f"{self.feats[i]}: {j:.3f}" for i, j in zip(self.reg_vars, self.params)]
        print(f"{' ' * const}   | Regression Params:")
        for p in param_format:
            print(f"{' ' * (const + width)}   | {p}")

    def print_tree(self):
        self._print_info()

        if self.left is not None:
            self.left.print_tree()

        if self.right is not None:
            self.right.print_tree()

        if self.left is None and self.right is None:
            self._print_params()

    def _find_params(self, row):
        if self.best_feat is not None:
            if row[self.best_feat] <= self.best_val:
                return self.left._find_params(row)
            else:
                return self.right._find_params(row)
        else:
            return (self.params, self.ID)

    def apply(self, X=None):
        X = self.X if X is None else X
        if hasattr(X, "values"):
            X = X.values
        N = X.shape[0]
        parms = []
        ids = []
        for i in range(N):
            parms_i, id_i = self._find_params(X[i, :])
            parms.append(parms_i)
            ids.append(id_i)
        return np.array(parms), np.array(ids)

    def predict(self, X=None):
        X = self.X if X is None else X
        if hasattr(X, "values"):
            X = X.values
        params, ids = self.apply(X)
        X = X[:, self.reg_vars]
        return (X * params).sum(axis=1)

    def _format_params_for_graph(self):
        min_width = 0
        for feat_id in self.reg_vars:
            width = len(self.feats[feat_id]) + 12
            if width > min_width:
                min_width = width

        param_format = [
            f"{i}: {j:5.3f}".rjust(min_width) for i, j in zip(self.feats, self.params)
        ]
        return param_format

    @staticmethod
    def _hex_to_rgb(value):
        value = value.lstrip("#")
        lv = len(value)
        return tuple(int(value[i: i + lv // 3], 16) for i in range(0, lv, lv // 3))

    @staticmethod
    def _rgb_to_hex(rgb):
        return "#%02x%02x%02x" % rgb

    def _build_tree_graph(self, tree=None):
        tree = Tree() if tree is None else tree

        if self.rule is None:
            param_format = self._format_params_for_graph()
            rule = "\n".join(param_format)
        else:
            rule = self.rule

        mse_fmt = f"mse = {self.mse:.3f}\n"
        pct_smp = f"samples = {self.N / TreeComboLR.init_N * 100:0.1f}%\n"
        rule = mse_fmt + pct_smp + rule

        if self.ID == 0:
            tree.create_node(rule, self.ID, data=self.mse)  # root
        else:
            tree.create_node(rule, self.ID, parent=self.parent, data=self.mse)

        if self.left is not None:
            self.left._build_tree_graph(tree)

        if self.right is not None:
            self.right._build_tree_graph(tree)

        return tree

    def to_graphviz(self, filename=None, shape="rectangle", graph="digraph"):
        # adapted from treelib.tree implementation
        """Exports the tree in the dot format of the graphviz software"""
        tree = self._build_tree_graph()

        # coloring of nodes
        min_col = "#FFFFFF"
        max_col = "#E58139"
        min_rgb = self._hex_to_rgb(min_col)
        max_rgb = self._hex_to_rgb(max_col)

        # interpolate between max and min
        rinterp = interp1d(
            [TreeComboLR.min_mse, TreeComboLR.max_mse],
            [min_rgb[0], max_rgb[0]],
        )
        ginterp = interp1d(
            [TreeComboLR.min_mse, TreeComboLR.max_mse],
            [min_rgb[1], max_rgb[1]],
        )
        binterp = interp1d(
            [TreeComboLR.min_mse, TreeComboLR.max_mse],
            [min_rgb[2], max_rgb[2]],
        )

        # get information for graph output
        nodes, connections = [], []
        if tree.nodes:
            for n in tree.expand_tree(mode=tree.WIDTH):
                nid = tree[n].identifier
                mse = [tree[n].data]
                rgb = (int(rinterp(mse)[0]), int(ginterp(mse)[0]), int(binterp(mse)[0]))
                myhex = self._rgb_to_hex(rgb)
                state = f'"{nid}" [label="{tree[n].tag}", fillcolor="{myhex}"]'
                nodes.append(state)

                for c in tree.children(nid):
                    cid = c.identifier
                    connections.append('"{0}" -> "{1}"'.format(nid, cid))

        # write nodes and connections to dot format
        is_plain_file = filename is not None
        if is_plain_file:
            f = codecs.open(filename, "w", "utf-8")
        else:
            f = StringIO()

        # format for graph
        node_style = [
            f"shape={shape}",
            'style="filled, rounded"',
            'color="black"',
            "fontname=helvetica",
        ]
        edge_style = ["fontname=helvetica"]
        f.write(graph + " tree {\n")
        f.write(f'node [{", ".join(node_style)}] ;\n')
        f.write(f'edge [{", ".join(edge_style)}] ;\n')

        for n in nodes:
            f.write("\t" + n + "\n")

        if len(connections) > 0:
            f.write("\n")

        for c in connections:
            f.write("\t" + c + "\n")

        f.write("}")

        if not is_plain_file:
            print(f.getvalue())

        f.close()
